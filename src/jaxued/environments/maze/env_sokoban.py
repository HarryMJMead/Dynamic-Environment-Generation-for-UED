from typing import Tuple
from enum import IntEnum
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from jaxued.environments import UnderspecifiedEnv
from .level import ObservedLevel, prefabs
from .env import EnvState, Observation, EnvParams, Maze, make_maze_map

class Actions(IntEnum):
    left = 0    # Turn left
    right = 1   # Turn right
    forward = 2 # Move forward
    use = 3  # Pick up an object
    reset = 4    # Drop an object
    # toggle = 5  # Toggle/activate an object
    # done = 6    # Done completing task

OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

COLORS = {
    'red'   : jnp.array([255, 0, 0]),
    'green' : jnp.array([0, 255, 0]),
    'blue'  : jnp.array([0, 0, 255]),
    'purple': jnp.array([112, 39, 195]),
    'yellow': jnp.array([255, 255, 0]),
    'grey'  : jnp.array([100, 100, 100]),
}

COLOR_TO_INDEX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    (1, 0), # right
    (0, 1), # down
    (-1, 0), # left
    (0, -1), # up
], dtype=jnp.int8)
    
class SokobanMaze(Maze):
    def __init__(
        self,
        max_height=13,
        max_width=13,
        agent_view_size=5,
        see_agent = False,
        normalize_obs = False,
        fully_obs = False,
        penalize_time = True,
        key_reward = 0.0,
    ):
        super().__init__(
            max_height=max_height,
            max_width=max_width,
            agent_view_size=agent_view_size,
            see_agent=see_agent,
            normalize_obs=normalize_obs,
            fully_obs=fully_obs,
            penalize_time=penalize_time,
            key_reward=key_reward,
        )
        self.show_boxes = True

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def init_state_from_level(self, level: ObservedLevel) -> EnvState:
        return super().init_state_from_level(level).replace(observation_map=level.observation_map, 
                                                            box_map_reset=jnp.array(level.box_map, dtype=jnp.bool_), 
                                                            agent_pos_reset=jnp.array(level.agent_pos, dtype=jnp.uint32),
                                                            agent_dir_reset=jnp.array(level.agent_dir, dtype=jnp.uint8))

    def update_state_from_level(self, level, state):
        update_level = level.replace(box_map=jnp.where(state.observation_map, state.box_map, level.box_map))
        obs, state = super().update_state_from_level(update_level, state)
        return obs, state.replace(observation_map=level.observation_map,
                                  box_map_reset=jnp.array(level.box_map, dtype=jnp.bool_),
                                  agent_pos_reset=jnp.array(level.agent_pos, dtype=jnp.uint32),
                                  agent_dir_reset=jnp.array(level.agent_dir, dtype=jnp.uint8))

    def _step_agent(self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams) -> Tuple[EnvState, float]:
        # Update agent position (forward action)
        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + (action == Actions.forward)*DIR_TO_VEC[state.agent_dir], 0), 
            jnp.array([self.max_width-1, self.max_height-1], dtype=jnp.uint32)
        )

        box_fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + 2*(action == Actions.forward)*DIR_TO_VEC[state.agent_dir], 0), 
            jnp.array([self.max_width-1, self.max_height-1], dtype=jnp.uint32)
        )

        # Check for key
        fwd_pos_has_blocked_door = jnp.logical_and(jnp.logical_and(fwd_pos[0] == state.door_pos[0], fwd_pos[1] == state.door_pos[1]), state.door_placed==1)
        fwd_pos_has_key = jnp.logical_and(jnp.logical_and(fwd_pos[0] == state.key_pos[0], fwd_pos[1] == state.key_pos[1]), state.key_placed)
        agent_has_key = jnp.logical_or(state.has_key, fwd_pos_has_key)
        key_placed = state.key_placed * (1 - agent_has_key) + agent_has_key*2

        # Can't go past wall or goal
        fwd_pos_has_wall = state.wall_map[fwd_pos[1], fwd_pos[0]]
        fwd_pos_has_goal = jnp.logical_and(jnp.logical_and(fwd_pos[0] == state.goal_pos[0], fwd_pos[1] == state.goal_pos[1]), state.goal_placed)
        fwd_pos_blocked = jnp.logical_or(jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal), fwd_pos_has_blocked_door)
        
        # Check for box
        fwd_pos_has_box = state.box_map[fwd_pos[1], fwd_pos[0]]

        # Check if box can be pushed
        box_fwd_pos_has_box = state.box_map[box_fwd_pos[1], box_fwd_pos[0]]
        box_fwd_pos_has_wall = state.wall_map[box_fwd_pos[1], box_fwd_pos[0]]
        box_fwd_pos_has_goal = jnp.logical_and(jnp.logical_and(box_fwd_pos[0] == state.goal_pos[0], box_fwd_pos[1] == state.goal_pos[1]), state.goal_placed)
        box_fwd_pos_has_door = jnp.logical_and(jnp.logical_and(box_fwd_pos[0] == state.door_pos[0], box_fwd_pos[1] == state.door_pos[1]), state.door_placed)
        box_fwd_pos_has_key = jnp.logical_and(jnp.logical_and(box_fwd_pos[0] == state.key_pos[0], box_fwd_pos[1] == state.key_pos[1]), state.key_placed)

        box_fwd_pos_blocked = jnp.any(jnp.array([
            box_fwd_pos_has_box,
            box_fwd_pos_has_wall,
            box_fwd_pos_has_goal,
            box_fwd_pos_has_door,
            box_fwd_pos_has_key
        ]))

        box_move_blocked = jnp.logical_and(fwd_pos_has_box, box_fwd_pos_blocked)
        move_box = jnp.logical_and(fwd_pos_has_box, ~box_fwd_pos_blocked)

        moved_box_map = state.box_map.at[box_fwd_pos[1], box_fwd_pos[0]].set(True).at[fwd_pos[1], fwd_pos[0]].set(False)
        box_map = jax.lax.select(move_box, moved_box_map, state.box_map)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_blocked, box_move_blocked)
        agent_pos = (fwd_pos_blocked*state.agent_pos + (~fwd_pos_blocked)*fwd_pos).astype(jnp.uint32)

        # Update agent direction (left_turn or right_turn action)
        agent_dir_offset = 0 + (action == Actions.right) - (action == Actions.left)
        agent_dir = (state.agent_dir + agent_dir_offset) % 4

        # Check Door Unlock
        fwd_pos_door = jnp.minimum(
            jnp.maximum(state.agent_pos + DIR_TO_VEC[state.agent_dir], 0), 
            jnp.array([self.max_width-1, self.max_height-1], dtype=jnp.uint32)
        )
        fwd_pos_has_door = jnp.logical_and(jnp.logical_and(fwd_pos_door[0] == state.door_pos[0], fwd_pos_door[1] == state.door_pos[1]), state.door_placed==1)
        unlock_door = jnp.logical_and(jnp.logical_and(fwd_pos_has_door, action == Actions.use), agent_has_key)
        door_placed = state.door_placed * (1 - unlock_door) + 2*unlock_door

        boxes_remaining = jnp.logical_xor(box_map, state.box_goal_map).any()
        boxes_finished = jnp.logical_and(box_map, state.box_goal_map).sum()
        box_complete = jnp.logical_and(boxes_remaining == 0, boxes_finished >= jnp.maximum(state.max_boxes, 1))
        
        level_finished = jnp.logical_or(box_complete, fwd_pos_has_goal)

        # Reset Level
        agent_pos = jax.lax.select(action == Actions.reset, state.agent_pos_reset, agent_pos)
        agent_dir = jax.lax.select(action == Actions.reset, state.agent_dir_reset, agent_dir)
        box_map = jax.lax.select(action == Actions.reset, state.box_map_reset, box_map)


        if self.penalize_time:
            reward = (1.0 - 0.9*((state.time+1)/params.max_steps_in_episode))*level_finished
        else:
            reward = jax.lax.select(level_finished, 1., 0.)
        
        if self.key_reward > 0:
            reward = reward + self.key_reward * jnp.logical_and(key_placed == 2, state.key_placed == 1) + self.key_reward * jnp.logical_and(door_placed == 2, state.door_placed == 1)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir=agent_dir,  
                terminal=level_finished,
                has_key=agent_has_key,
                key_placed=key_placed,
                door_placed=door_placed,
                box_map=box_map),
            reward
        )