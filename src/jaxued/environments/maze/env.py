from typing import Tuple
from enum import IntEnum
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from jaxued.environments import UnderspecifiedEnv
from .level import Level, prefabs

class Actions(IntEnum):
    left = 0    # Turn left
    right = 1   # Turn right
    forward = 2 # Move forward
    use = 3  # Pick up an object
    # drop = 4    # Drop an object
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

@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    agent_dir: int
    goal_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool
    goal_placed: chex.Array

    # Door and Key Requirements
    has_key: chex.Array = jnp.array(False, dtype=jnp.bool_)
    key_pos: chex.Array = jnp.array([0, 0], dtype=jnp.uint32)
    key_placed: chex.Array = jnp.array(0, dtype=jnp.uint8)
    door_pos: chex.Array = jnp.array([0, 0], dtype=jnp.uint32)
    door_placed: chex.Array = jnp.array(0, dtype=jnp.uint8)
    
    # Sokoban Requirements
    box_map: chex.Array = jnp.array([[0]], dtype=jnp.bool_)
    box_goal_map: chex.Array = jnp.array([[0]], dtype=jnp.bool_)
    observation_map: chex.Array = jnp.array([[0]], dtype=jnp.uint8)
    max_boxes: int = 0
    box_map_reset: chex.Array = jnp.array([[0]], dtype=jnp.bool_)
    agent_pos_reset: chex.Array = jnp.array([0, 0], dtype=jnp.uint32)
    agent_dir_reset: chex.Array = jnp.array(0, dtype=jnp.uint8)

@struct.dataclass
class Observation:
    image: chex.Array
    agent_dir: int
    obs_location: chex.Array = None
    agent_observation: chex.Array = None
    agent_location: chex.Array = None
    has_key: chex.Array = None


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 250
    
class Maze(UnderspecifiedEnv):
    """This is an implementation of a Maze in a minigrid-style environment.

    Args:
        max_height (int, optional): The maximum height. Levels themselves can be smaller than this. Defaults to 13.
        max_width (int, optional): The maximum width. Defaults to 13.
        agent_view_size (int, optional): The number of tiles the agent can see in front of itself. Defaults to 5.
        see_agent (bool, optional): If this is true, the agent's observation includes the agent. By default this is false, which is fine because the observation is egocentric, so the agent is always at the same position in the observation. Defaults to False.
        normalize_obs (bool, optional): If true, divides the observations by 10 to normalize them. Defaults to False.
        fully_obs (bool, optional): If this is true, the agent sees the entire grid. Defaults to False.
        penalize_time (bool, optional): If this is true, the reward for obtaining the goal decreases at every tim. Defaults to True.
    """
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
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.agent_view_size = agent_view_size
        self.see_agent = see_agent
        self.normalize_obs = normalize_obs
        self.fully_obs = fully_obs
        self.penalize_time = penalize_time
        self.key_reward = key_reward
        self.show_boxes = False

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        state, reward = self._step_agent(rng, state, action, params)
        maze_map = make_maze_map(state, self.agent_view_size-1, show_boxes=self.show_boxes)
        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1, maze_map=maze_map)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return (
            self.get_obs(state),
            state,
            reward.astype(jnp.float32),
            done,
            {},
        )
        
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = self.init_state_from_level(level)
        return self.get_obs(state), state

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    # ===
    
    def init_state_from_level(self, level: Level) -> EnvState:
        maze_map = make_maze_map(level, self.agent_view_size-1, show_boxes=self.show_boxes)
        return EnvState(
            agent_pos=jnp.array(level.agent_pos, dtype=jnp.uint32),
            agent_dir=jnp.array(level.agent_dir, dtype=jnp.uint8),
            goal_pos=jnp.array(level.goal_pos, dtype=jnp.uint32),
            goal_placed=jnp.array(level.goal_placed, dtype=jnp.bool_),
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            box_map=jnp.array(level.box_map, dtype=jnp.bool_),
            box_goal_map=jnp.array(level.box_goal_map, dtype=jnp.bool_),
            max_boxes=jnp.array(level.max_boxes, dtype=jnp.uint32),
            maze_map=maze_map,
            time=0,
            terminal=False,
            door_pos=jnp.array(level.door_pos, dtype=jnp.uint32),
            door_placed=jnp.array(level.door_placed, dtype=jnp.uint8),
            key_pos=jnp.array(level.key_pos, dtype=jnp.uint32),
            key_placed=jnp.array(level.key_placed, dtype=jnp.uint8),
        )
    
    def update_state_from_level(self, level: Level, state: EnvState) -> EnvState:
        maze_map = make_maze_map(level, self.agent_view_size-1, show_boxes=self.show_boxes)
        state = state.replace(
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            box_map=jnp.array(level.box_map, dtype=jnp.bool_),
            box_goal_map=jnp.array(level.box_goal_map, dtype=jnp.bool_),
            max_boxes=jnp.array(level.max_boxes, dtype=jnp.uint32),
            maze_map=maze_map,
            goal_pos=jnp.array(level.goal_pos, dtype=jnp.uint32),
            goal_placed=jnp.array(level.goal_placed, dtype=jnp.bool_),
            
            door_pos=jnp.array(level.door_pos, dtype=jnp.uint32),
            door_placed=jnp.array(jnp.maximum(state.door_placed, level.door_placed), dtype=jnp.uint8),
            key_pos=jnp.array(level.key_pos, dtype=jnp.uint32),
            key_placed=jnp.array(jnp.maximum(state.key_placed, level.key_placed), dtype=jnp.uint8),
        )
        return self.get_obs(state), state

    def update_state_from_adv_state(self, adv_state, state: EnvState, index: int) -> EnvState:
        agent_pos = jax.lax.dynamic_index_in_dim(adv_state.agent_locs, index, keepdims=False)
        agent_dir = jax.lax.dynamic_index_in_dim(adv_state.agent_dirs, index, keepdims=False)
        state = state.replace(
            agent_pos=agent_pos,
            agent_dir=agent_dir
        )
        return self.update_state_from_level(adv_state.level, state)

    def get_obs(self, state: EnvState) -> Observation:
        if self.fully_obs:
            return self._get_full_obs(state)
        return self._get_partial_obs(state)
    
    def get_goal_location(self, state: EnvState) -> chex.Array:
        y, x = state.goal_pos.astype(jnp.int32)
        goal_observation = jnp.zeros((state.wall_map.shape[0], state.wall_map.shape[1]))
        return jax.lax.select(state.goal_placed, jax.lax.dynamic_update_slice(goal_observation, jnp.ones((1, 1))*10, (x, y)), goal_observation)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_steps, state.terminal)
        
    def _get_full_obs(self, state: EnvState) -> Observation:
        """Return coomplete grid view"""
        padding = self.agent_view_size-1
        obs = jax.lax.dynamic_slice(state.maze_map, (padding, padding, 0), (self.max_height, self.max_width, 3))
        obs = obs.at[state.agent_pos[1], state.agent_pos[0]].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.agent_dir], dtype=jnp.uint8)
        )
        image = obs.astype(jnp.uint8)
        if self.normalize_obs:
            image = image/10.0

        agent_location = jnp.concatenate([state.agent_dir.reshape((1,)), state.agent_pos])

        return Observation(image=image, agent_dir=state.agent_dir, agent_location=agent_location)
        
    def _get_partial_obs(self, state: EnvState) -> Observation:
        """Return limited grid view ahead of agent."""
        dir_vec = DIR_TO_VEC[state.agent_dir]
        
        obs_fwd_bound1 = state.agent_pos
        obs_fwd_bound2 = state.agent_pos + dir_vec*(self.agent_view_size-1)

        side_offset = self.agent_view_size//2
        obs_side_bound1 = state.agent_pos + (dir_vec == 0)*side_offset
        obs_side_bound2 = state.agent_pos - (dir_vec == 0)*side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])

        # Clip obs to grid bounds appropriately
        padding = self.agent_view_size-1
        xmin, ymin = jnp.min(all_bounds, 0) + padding
        obs = jax.lax.dynamic_slice(state.maze_map, (ymin, xmin, 0), (self.agent_view_size, self.agent_view_size, 3))

        # obs_location = jnp.zeros_like(state.maze_map)
        # obs_location = jax.lax.dynamic_update_slice(obs_location, obs, (ymin, xmin, 0))

        obs_location = jnp.zeros((state.maze_map.shape[0], state.maze_map.shape[1], 1), dtype=jnp.uint8)
        obs_location = jax.lax.dynamic_update_slice(obs_location, jnp.ones((self.agent_view_size, self.agent_view_size, 1), dtype=jnp.uint8), (ymin, xmin, 0))

        obs = (state.agent_dir == 0)*jnp.rot90(obs, 1) + \
              (state.agent_dir == 1)*jnp.rot90(obs, 2) + \
              (state.agent_dir == 2)*jnp.rot90(obs, 3) + \
              (state.agent_dir == 3)*jnp.rot90(obs, 4)

        if self.see_agent:
            obs = obs.at[-1, side_offset].set(
                jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.agent_dir], dtype=jnp.uint8)
            )

        image = obs.astype(jnp.uint8)
        if self.normalize_obs:
            image = image/10.0

        y, x = state.agent_pos.astype(jnp.int32)
        agent_observation = obs_location[padding:-padding, padding:-padding].squeeze().astype(jnp.float32)
        agent_observation = jax.lax.dynamic_update_slice(agent_observation, jnp.ones((1, 1))*10, (x, y))

        agent_location = jnp.concatenate([state.agent_dir.reshape((1,)), state.agent_pos])

        return Observation(image=image.transpose(1, 0, 2), agent_dir=state.agent_dir, obs_location=obs_location, agent_observation=agent_observation, agent_location=agent_location, has_key=state.has_key)
        
    def _step_agent(self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams) -> Tuple[EnvState, float]:
        # Update agent position (forward action)
        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + (action == Actions.forward)*DIR_TO_VEC[state.agent_dir], 0), 
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

        if self.penalize_time:
            reward = (1.0 - 0.9*((state.time+1)/params.max_steps_in_episode))*fwd_pos_has_goal
        else:
            reward = jax.lax.select(fwd_pos_has_goal, 1., 0.)
        
        if self.key_reward > 0:
            reward = reward + self.key_reward * jnp.logical_and(key_placed == 2, state.key_placed == 1) + self.key_reward * jnp.logical_and(door_placed == 2, state.door_placed == 1)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir=agent_dir,  
                terminal=fwd_pos_has_goal,
                has_key=agent_has_key,
                key_placed=key_placed,
                door_placed=door_placed),
            reward
        )
        
def make_maze_map(
    level: Level,
    padding=0,
    ignore_goal=False,
    show_boxes=False,
) -> chex.Array:
    """This function is used internally in this class, and it creates a single array map of the entire level. This includes walls, the goal but not the agent.

    Args:
        level (Level): 
        padding (int, optional): How much to pad the level with, which is used to ensure the agent's observations when it is near the edge is well-defined. Defaults to 0.
        ignore_goal (bool, optional): If true, the observation does not contain the goal. Defaults to False.

    Returns:
        chex.Array: The full maze map.
    """
    # Expand maze map to H x W x C
    empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
    wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
    maze_map = jnp.array(jnp.expand_dims(level.wall_map, -1), dtype=jnp.uint8)
    maze_map = jnp.where(maze_map > 0, wall, empty)

    box = jnp.array([OBJECT_TO_INDEX['box'], COLOR_TO_INDEX['yellow'], 0], dtype=jnp.uint8)
    box_goal = jnp.array([OBJECT_TO_INDEX['box'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)
    box_complete = jnp.array([OBJECT_TO_INDEX['box'], COLOR_TO_INDEX['purple'], 0], dtype=jnp.uint8)
    if show_boxes:
        box_map = jnp.array(jnp.expand_dims(level.box_map, -1), dtype=jnp.uint8)
        box_goal_map = jnp.array(jnp.expand_dims(level.box_goal_map, -1), dtype=jnp.uint8)
        box_complete_map = jnp.array(jnp.expand_dims(jnp.logical_and(level.box_map, level.box_goal_map), -1), dtype=jnp.uint8)
        maze_map = jnp.where(box_map > 0, box, maze_map)
        maze_map = jnp.where(box_goal_map > 0, box_goal, maze_map)
        maze_map = jnp.where(box_complete_map > 0, box_complete, maze_map)

    goal = jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)
    goal_x,goal_y = level.goal_pos
    ignore_goal = jnp.logical_or(~level.goal_placed, ignore_goal)
    maze_map = jax.lax.select(ignore_goal, maze_map, maze_map.at[goal_y,goal_x,:].set(goal))

    door = jnp.array([OBJECT_TO_INDEX['door'], level.door_placed, 0], dtype=jnp.uint8)
    door_x,door_y = level.door_pos
    maze_map = jax.lax.select(level.door_placed == 0, maze_map, maze_map.at[door_y,door_x,:].set(door))

    key = jnp.array([OBJECT_TO_INDEX['key'], level.key_placed, 0], dtype=jnp.uint8)
    key_x,key_y = level.key_pos
    maze_map = jax.lax.select(level.key_placed == 0, maze_map, maze_map.at[key_y,key_x,:].set(key))

    if padding > 0:
        maze_map_padded = jnp.tile(wall.reshape((1, 1, *empty.shape)), (maze_map.shape[0]+2*padding, maze_map.shape[1]+2*padding, 1))
        maze_map_padded = maze_map_padded.at[padding:-padding,padding:-padding,:].set(maze_map)

        # Add surrounding walls
        wall_start = padding-1 # start index for walls
        wall_end_y = maze_map_padded.shape[0] - wall_start - 1
        wall_end_x = maze_map_padded.shape[1] - wall_start - 1
        maze_map_padded = maze_map_padded.at[wall_start,wall_start:wall_end_x+1,:].set(wall) # top
        maze_map_padded = maze_map_padded.at[wall_end_y,wall_start:wall_end_x+1,:].set(wall) # bottom
        maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_start,:].set(wall) # left
        maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_end_x,:].set(wall) # right

        return maze_map_padded
    else:
        return maze_map