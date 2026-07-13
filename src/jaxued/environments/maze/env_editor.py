from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from .level import Level, ObservedLevel
from jaxued.environments import UnderspecifiedEnv
from .env import COLOR_TO_INDEX, OBJECT_TO_INDEX, Maze, make_maze_map

class MultiDiscrete(spaces.Space):
    """Minimal jittable class for multidiscrete gymnax spaces."""

    def __init__(self, num_cat_vec: Tuple[int]):
        # Ensure all dimensions are non-negative integers
        self.n = jnp.array(num_cat_vec)
        assert jnp.all(self.n > 0)
        self.shape = self.n.shape
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action from multidiscrete space."""
        # Generate random integers for each dimension independently
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        """Check whether a given action is within the multidiscrete space."""
        lower_bound = jnp.zeros_like(self.n)
        upper_bound = self.n
        return jnp.all((x >= lower_bound) & (x < upper_bound))


@struct.dataclass
class EnvState:
    level: ObservedLevel
    time: int
    terminal: bool
    agent_locs: chex.Array = None
    agent_dirs: chex.Array = None
    box_locs: chex.Array = None
    has_reset: bool = False

@struct.dataclass
class Observation:
    image: chex.Array
    observation_map: chex.Array
    action_mask: chex.Array
    agent_observations: chex.Array
    agent_steps: chex.Array
    time: int
    random_z: chex.Array
    place_goal: chex.Array = None
    goal_placed: chex.Array = None
    key_placed: chex.Array = None
    door_placed: chex.Array = None
    agent_dirs: chex.Array = None
    agent_values: chex.Array = None
    agent_boxes: chex.Array = None
    box_count: int = 0
    box_goal_count: int = 0
    max_boxes: int = 0
    
@struct.dataclass
class EnvParams:
    pass

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    (1, 0), # right
    (0, 1), # down
    (-1, 0), # left
    (0, -1), # up
], dtype=jnp.int8)
    
class MazeEditor(UnderspecifiedEnv):
    """
        This environment allows the adversary to generate a level. The adversary can move the goal, move the agent, rotate the agent, or toggle walls.
        The action space is discrete, of dimension w*h, where w and h are the width and height of the maze, respectively. The first action moves the goal, the second action rotates the agent, the third action moves the agent, and the fourth action onwards toggles walls.
    """
    def __init__(self, env: Maze, random_z_dimensions: int = 16, zero_out_random_z: bool = False, num_agents = 2, agent_view_size = 5, set_start = False, set_init_pos = True):
        super().__init__()
        self._env = env
        self.random_z_dimensions = random_z_dimensions
        self.zero_out_random_z = zero_out_random_z
        self.num_agents = num_agents
        self.agent_view_size = agent_view_size
        self.set_start = set_start
        self.set_init_pos = set_init_pos
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
        # Do not edit level if in terminal state
        rng, rng_obs = jax.random.split(rng)
        new_level = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(state.terminal, x, y),
            state.level,
            self._edit_level(rng, state, action, params)
        )
        # Check game condition & no. steps for termination condition
        state = state.replace(level=new_level, time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return self.get_obs(rng_obs, state), state, 0, done, {}
        
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = self.init_state_from_level(level)
        return self.get_obs(rng, state), state
    

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)
    
    # ===

    @property
    def num_actions(self) -> int:
        return self._env.max_width * self._env.max_height

    def get_obs(self, rng: chex.Array, state: EnvState):
        goal_idx = state.level.goal_pos[1] * self._env.max_width + state.level.goal_pos[0]
        agent_idx = state.level.agent_pos[1] * self._env.max_width + state.level.agent_pos[0]
        
        #action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*2)
        
        maze_map = make_maze_map(state.level, ignore_goal=state.time==0, show_boxes=self.show_boxes)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1], state.level.agent_pos[0]].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
        maze_map = jax.lax.select(state.time > 2, maze_map_with_agent, maze_map)
    
        return Observation(
            image=maze_map,
            observation_map = None,
            action_mask=None,
            agent_observations=jnp.ones((self._env.max_width, self._env.max_height, self.num_agents), dtype=jnp.float32),
            agent_steps = jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return False
        
    def init_state_from_level(self, level):
        return EnvState(
            level=level,
            time=jnp.array(0, dtype=jnp.uint32),
            terminal=False,
            agent_locs=jnp.tile(level.agent_pos, (self.num_agents, 1)),
            agent_dirs=jnp.tile(level.agent_dir, (self.num_agents)),
            box_locs=jnp.tile(level.box_map, (self.num_agents, 1, 1)),
            has_reset=False,
        )
        
    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        def move_goal():
            level = state.level
            x, y = edit_idx % max_w, edit_idx // max_w
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), goal_placed=jnp.array(True, dtype=jnp.bool_))
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False)
            new_edit_idx = jax.lax.select(
                edit_idx == goal_idx,
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_idx,
            )
            x, y = new_edit_idx % max_w, new_edit_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32))
        
        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            x, y = edit_idx % max_w, edit_idx // max_w
            wall_val = jax.lax.select(
                (edit_idx == goal_idx) | (edit_idx == agent_idx),
                False,
                ~level.wall_map[y, x]
            )

            return level.replace(wall_map=level.wall_map.at[y, x].set(wall_val))
        
        if self.set_init_pos:
            edit_action = state.time.clip(None, 3)
            level = jax.lax.switch(edit_action, [move_goal, rotate_agent, move_agent, toggle_wall])
        else:
            edit_action = state.time.clip(None, 1)
            level = jax.lax.switch(edit_action, [move_goal, toggle_wall])
        
        return level

class KeyMazeEditor(MazeEditor):
    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        def move_goal():
            level = state.level
            x, y = edit_idx % max_w, edit_idx // max_w
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), goal_placed=jnp.array(True, dtype=jnp.bool_))
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False)
            new_edit_idx = jax.lax.select(
                edit_idx == goal_idx,
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_idx,
            )
            x, y = new_edit_idx % max_w, new_edit_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32))
        
        def move_door():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False).at[agent_idx].set(False)
            new_edit_idx = jax.lax.select(
                jnp.logical_or(edit_idx == goal_idx, edit_idx == agent_idx),
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_idx,
            )
            x, y = new_edit_idx % max_w, new_edit_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), door_pos=jnp.array([x, y], dtype=jnp.uint32), door_placed=jnp.array(1, dtype=jnp.uint8))
        
        def move_key():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            door_idx = level.door_pos[1] * max_w + level.door_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False).at[agent_idx].set(False).at[door_idx].set(False)
            new_edit_idx = jax.lax.select(
                jnp.logical_or(jnp.logical_or(edit_idx == goal_idx, edit_idx == agent_idx), edit_idx == door_idx),
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_idx,
            )
            x, y = new_edit_idx % max_w, new_edit_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), key_pos=jnp.array([x, y], dtype=jnp.uint32), key_placed=jnp.array(1, dtype=jnp.uint8))

        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            x, y = edit_idx % max_w, edit_idx // max_w
            wall_val = jax.lax.select(
                (edit_idx == goal_idx) | (edit_idx == agent_idx),
                False,
                ~level.wall_map[y, x]
            )

            return level.replace(wall_map=level.wall_map.at[y, x].set(wall_val))
        
        if self.set_init_pos:
            edit_action = state.time.clip(None, 5)
            level = jax.lax.switch(edit_action, [move_goal, rotate_agent, move_agent, move_door, move_key, toggle_wall])
        else:
            edit_action = state.time.clip(None, 3)
            level = jax.lax.switch(edit_action, [move_goal, move_door, move_key, toggle_wall])
        
        return level


class ObservedMazeEditor(MazeEditor):

    @property
    def num_actions(self) -> int:
        return self._env.max_width * self._env.max_height * 2

    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        edit_loc_idx = edit_idx % (max_w * max_h)
        edit_type_idx = edit_idx // (max_w * max_h)
        def move_goal():
            level = state.level
            x, y = edit_loc_idx % max_w, edit_loc_idx // max_w
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), goal_placed=jnp.array(True, dtype=jnp.bool_))
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False)
            new_edit_loc_idx = jax.lax.select(
                edit_loc_idx == goal_idx,
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_loc_idx,
            )
            x, y = new_edit_loc_idx % max_w, new_edit_loc_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True))
        
        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            x, y = edit_loc_idx % max_w, edit_loc_idx % (max_w * max_h) // max_w
            wall_val = jax.lax.select(
                (edit_loc_idx == goal_idx) | (edit_loc_idx == agent_idx),
                False,
                edit_type_idx != 0
            )

            return level.replace(wall_map=level.wall_map.at[y, x].set(wall_val), observation_map=level.observation_map.at[y, x].set(True))
        
        edit_action = state.time.clip(None, 3)
        level = jax.lax.switch(edit_action, [move_goal, rotate_agent, move_agent, toggle_wall])
        
        return level


class ObservedMazeEditorWithGoal(MazeEditor):

    @property
    def num_actions(self) -> int:
        return self._env.max_width * self._env.max_height * 3

    def get_obs(self, rng: chex.Array, state: EnvState):
        goal_idx = state.level.goal_pos[1] * self._env.max_width + state.level.goal_pos[0]
        agent_idx = state.level.agent_pos[1] * self._env.max_width + state.level.agent_pos[0]
        
        action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*2 + [jnp.logical_and(~state.level.observation_map.flatten(), ~state.level.goal_placed)])
        #action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*3)
        
        maze_map = make_maze_map(state.level, ignore_goal=state.time==0, show_boxes=self.show_boxes)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1], state.level.agent_pos[0]].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
        maze_map = jax.lax.select(state.time > 0, maze_map_with_agent, maze_map)
    
        return Observation(
            image=maze_map,
            observation_map = state.level.observation_map,
            action_mask=action_mask,
            agent_observations=jnp.ones((self._env.max_width, self._env.max_height, self.num_agents)),
            agent_steps = jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
            place_goal=jnp.array(True, dtype=jnp.bool_)
        )

    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        edit_loc_idx = edit_idx % (max_w * max_h)
        edit_type_idx = edit_idx // (max_w * max_h)

        def move_goal():
            level = state.level
            x, y = edit_loc_idx % max_w, edit_loc_idx // max_w
            return jax.lax.cond(
                level.goal_placed,
                lambda: level,
                lambda : level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), goal_placed=jnp.array(True, dtype=jnp.bool_))
            )
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent():
            level = state.level
            
            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False)
            new_edit_loc_idx = jax.lax.select(
                edit_loc_idx == goal_idx,
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_loc_idx,
            )
            x, y = new_edit_loc_idx % max_w, new_edit_loc_idx // max_w
            
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True))
        
        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            x, y = edit_loc_idx % max_w, edit_loc_idx % (max_w * max_h) // max_w
            wall_val = jax.lax.select(
                (edit_loc_idx == goal_idx) | (edit_loc_idx == agent_idx),
                False,
                edit_type_idx != 0
            )

            return level.replace(wall_map=level.wall_map.at[y, x].set(wall_val), observation_map=level.observation_map.at[y, x].set(True))
        
        edit_action = state.time.clip(None, 2)
        edit_goal = jnp.clip(edit_type_idx-1, 0, 1)
        level = jax.lax.switch(edit_action, [move_agent, rotate_agent, lambda : jax.lax.switch(edit_goal, [toggle_wall, move_goal])])
        
        return level


class LocalMazeEditor(MazeEditor):

    @property
    def num_actions(self) -> int:
        return self.num_agents * self.agent_view_size**2 * 3

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        # Do not edit level if in terminal state
        rng, rng_obs = jax.random.split(rng)
        new_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(state.terminal, x, y),
            state,
            self._edit_level(rng, state, action, params)
        )
        # Check game condition & no. steps for termination condition
        state = new_state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return self.get_obs(rng_obs, state), state, 0, done, {}
    
    def agent_placement_action_mask(self, time):
        max_w, max_h = self._env.max_width, self._env.max_height

        clipped_time = time.clip(None, 2)
        def direction():
            return jnp.zeros(self.num_actions, dtype=jnp.bool_).at[:4].set(True)
        def x_pos():
            return jnp.zeros(self.num_actions, dtype=jnp.bool_).at[:max_w].set(True)
        def y_pos():
            return jnp.zeros(self.num_actions, dtype=jnp.bool_).at[:max_h].set(True)

        return jax.lax.switch(clipped_time, [direction, x_pos, y_pos])

    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        dir_vec = DIR_TO_VEC[agent_dir]
        
        obs_fwd_bound1 = agent_pos
        obs_fwd_bound2 = agent_pos + dir_vec*(self.agent_view_size-1)

        side_offset = self.agent_view_size//2
        obs_side_bound1 = agent_pos + (dir_vec == 0)*side_offset
        obs_side_bound2 = agent_pos - (dir_vec == 0)*side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])
        padding = self.agent_view_size-1
        xmin, ymin = jnp.min(all_bounds, 0) + padding

        maze_map_w_cur_agent = maze_map.at[agent_pos[1] + self.agent_view_size-1, agent_pos[0] + self.agent_view_size-1].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['yellow'], agent_dir], dtype=jnp.uint8)
        )
        maze_map = jax.lax.select(include_agent, maze_map_w_cur_agent, maze_map)
        obs = jax.lax.dynamic_slice(maze_map, (ymin, xmin, 0), (self.agent_view_size, self.agent_view_size, 3))

        padded_observation_map = jnp.ones(maze_map.shape[0:2], dtype=jnp.bool_).at[padding:-padding, padding:-padding].set(level.observation_map)
        obs_map = jax.lax.dynamic_slice(padded_observation_map, (ymin, xmin), (self.agent_view_size, self.agent_view_size))

        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)])

        return obs, obs_map, action_mask

    def get_finished_obs(self, rng: chex.Array, state: EnvState, not_done: chex.Array):
        maze_map = make_maze_map(state.level, padding=self.agent_view_size-1, show_boxes=self.show_boxes)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1] + self.agent_view_size-1, state.level.agent_pos[0] + self.agent_view_size-1].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
        image, obs_map, action_mask = jax.vmap(self.get_agent_obs, in_axes=(0, 0, None, None))(state.agent_locs, state.agent_dirs, state.level, maze_map_with_agent)
        finished_obs, _, _ = self.get_agent_obs(state.level.goal_pos - jnp.array([2, 0]), jnp.array(0), state.level, maze_map_with_agent, False)
        finished_image = finished_obs[None, ...].repeat(self.num_agents, axis=0)
        image = jnp.where(not_done.reshape(-1, 1, 1, 1), image, finished_image)
        image = jnp.moveaxis(image, 0, -2).reshape(self.agent_view_size, self.agent_view_size, 3*self.num_agents)
        obs_map = jnp.moveaxis(obs_map, 0, -1)
        action_mask = jax.tree_util.tree_map(lambda x: x.flatten(), action_mask)

        if self.set_start:
            action_mask = jax.lax.select(state.time <= 2, self.agent_placement_action_mask(state.time), action_mask)
    
        return Observation(
            image=image/10,
            observation_map = obs_map,
            action_mask=action_mask,
            agent_observations=None,
            agent_steps = jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
            place_goal=jnp.array(True, dtype=jnp.bool_),
            goal_placed = jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            key_placed = jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            door_placed = jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            agent_dirs = state.agent_dirs,
            agent_values = jnp.zeros((self.num_agents,), dtype=jnp.float32),
        )

    def get_obs(self, rng: chex.Array, state: EnvState):
        goal_idx = state.level.goal_pos[1] * self._env.max_width + state.level.goal_pos[0]
        agent_idx = state.level.agent_pos[1] * self._env.max_width + state.level.agent_pos[0]
        
        #action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*2 + [jnp.logical_and(~state.level.observation_map.flatten(), ~state.level.goal_placed)])
        #action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*3)
        
        maze_map = make_maze_map(state.level, padding=self.agent_view_size-1, show_boxes=self.show_boxes)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1] + self.agent_view_size-1, state.level.agent_pos[0] + self.agent_view_size-1].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
        image, obs_map, action_mask = jax.vmap(self.get_agent_obs, in_axes=(0, 0, None, None))(state.agent_locs, state.agent_dirs, state.level, maze_map_with_agent)
        image = jnp.moveaxis(image, 0, -2).reshape(self.agent_view_size, self.agent_view_size, 3*self.num_agents)
        obs_map = jnp.moveaxis(obs_map, 0, -1)
        action_mask = jax.tree_util.tree_map(lambda x: x.flatten(), action_mask)

        if self.set_start:
            action_mask = jax.lax.select(state.time <= 2, self.agent_placement_action_mask(state.time), action_mask)
    
        return Observation(
            image=image/10,
            observation_map = obs_map,
            action_mask=action_mask,
            agent_observations=None,
            agent_steps = jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
            place_goal=jnp.array(True, dtype=jnp.bool_),
            goal_placed = jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            key_placed = jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            door_placed = jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            agent_dirs = state.agent_dirs,
            agent_values = jnp.zeros((self.num_agents,), dtype=jnp.float32),
        )

    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        edit_loc_idx = edit_idx % (self.agent_view_size**2)
        edit_type_idx = (edit_idx % (self.agent_view_size**2 * 3)) // self.agent_view_size**2
        edit_agent_idx = edit_idx // (self.agent_view_size**2 * 3)
        
        agent_pos = jax.lax.dynamic_index_in_dim(state.agent_locs, edit_agent_idx, axis=0).squeeze(0)
        agent_dir = jax.lax.dynamic_index_in_dim(state.agent_dirs, edit_agent_idx, axis=0).squeeze(0)

        dir_vec = DIR_TO_VEC[agent_dir]
        
        obs_fwd_bound1 = agent_pos
        obs_fwd_bound2 = agent_pos + dir_vec*(self.agent_view_size-1)

        side_offset = self.agent_view_size//2
        obs_side_bound1 = agent_pos + (dir_vec == 0)*side_offset
        obs_side_bound2 = agent_pos - (dir_vec == 0)*side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])
        xmin, ymin = jnp.min(all_bounds, 0)

        def move_goal():
            level = state.level
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin
            return jax.lax.cond(
                level.goal_placed,
                lambda: level,
                lambda : level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), goal_placed=jnp.array(True, dtype=jnp.bool_))
            )
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent_x():
            level = state.level
            x = edit_idx % max_w
            y = level.agent_pos[1]
            return level.replace(agent_pos=jnp.array([x, y], dtype=jnp.uint32))
        
        def move_agent_y():
            level = state.level
            x = level.agent_pos[0]
            y = edit_idx % max_h
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True))
        
        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin

            return level.replace(wall_map=level.wall_map.at[y, x].set(edit_type_idx != 0), observation_map=level.observation_map.at[y, x].set(True))

        def set_agent_locs():
            agent_locs=jnp.tile(level.agent_pos, (self.num_agents, 1))
            agent_dirs=jnp.tile(level.agent_dir, (self.num_agents))
            return state.replace(agent_locs=agent_locs, agent_dirs=agent_dirs)
        
        edit_goal = jnp.clip(edit_type_idx-1, 0, 1)
        edit_time = state.time.clip(None, 3)
        if self.set_start:
            level = jax.lax.switch(edit_time, [
                rotate_agent,
                move_agent_x,
                move_agent_y,
                lambda: jax.lax.switch(edit_goal, [toggle_wall, move_goal])
            ])

            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(edit_time <= 2, x, y),
                set_agent_locs(),
                state
            )
        else:
            level = jax.lax.switch(edit_goal, [toggle_wall, move_goal])
        
        return state.replace(level=level)
    
class LocalMazeEditorRotate(LocalMazeEditor):
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        def rotate(arr):
            return (agent_dir == 0)*jnp.rot90(arr, 1) + \
                   (agent_dir == 1)*jnp.rot90(arr, 2) + \
                   (agent_dir == 2)*jnp.rot90(arr, 3) + \
                   (agent_dir == 3)*jnp.rot90(arr, 4)
        
        obs, obs_map, _ = super().get_agent_obs(agent_pos, agent_dir, level, maze_map, include_agent)    
        obs, obs_map = jax.tree_util.tree_map(rotate, (obs, obs_map))
        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)])
        return obs, obs_map, action_mask
    
    def _edit_level(self, rng, state, edit_idx, params):
        edit_loc_idx = edit_idx % (self.agent_view_size**2)
        edit_agent_idx = edit_idx // (self.agent_view_size**2 * 3)
        
        agent_dir = jax.lax.dynamic_index_in_dim(state.agent_dirs, edit_agent_idx, axis=0).squeeze(0)

        def rot_index(idx):
            def rot_coords(x, y):
                x_r = (agent_dir == 0)*(self.agent_view_size - y - 1) + \
                    (agent_dir == 1)*(self.agent_view_size - x - 1) + \
                    (agent_dir == 2)*(y) + \
                    (agent_dir == 3)*(x)
                
                y_r = (agent_dir == 0)*(x) + \
                    (agent_dir == 1)*(self.agent_view_size - y - 1) + \
                    (agent_dir == 2)*(self.agent_view_size - x - 1) + \
                    (agent_dir == 3)*(y)
                return x_r, y_r
            
            x, y = idx % self.agent_view_size, idx // self.agent_view_size
            x_r, y_r = rot_coords(x, y)
            return x_r + y_r * self.agent_view_size
        
        r_edit_loc_idx = rot_index(edit_loc_idx)
        edit_idx = edit_idx + r_edit_loc_idx - edit_loc_idx
        return super()._edit_level(rng, state, edit_idx, params)

# Will only work for single agent - need to fix multi-agent action masking for action type selection
class LocalMazeEditorRotateSplitAct(LocalMazeEditorRotate):
    def action_space(self, params: EnvParams) -> MultiDiscrete:
        return MultiDiscrete(self.num_actions)

    @property
    def num_actions(self) -> int:
        return (self.num_agents * self.agent_view_size**2, 3)
    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        obs, obs_map, _ = super().get_agent_obs(agent_pos, agent_dir, level, maze_map, include_agent)    
        action_mask_loc = ~obs_map.flatten()
        action_mask_type = jnp.array([True, True, ~level.goal_placed])
        return obs, obs_map, (action_mask_loc, action_mask_type)
    
    def _edit_level(self, rng, state, edit_idxs, params):
        loc_idx, edit_type_idx = edit_idxs

        edit_loc_idx = loc_idx % (self.agent_view_size**2)
        edit_agent_idx = loc_idx // (self.agent_view_size**2)

        edit_idx = edit_loc_idx + (self.agent_view_size**2) * edit_type_idx + (self.agent_view_size**2 * 3) * edit_agent_idx
        
        return super()._edit_level(rng, state, edit_idx, params)
    
    def _reduce_action_mask(self, obs: Observation):
        action_mask_loc, action_mask_type = obs.action_mask
        action_mask_type = action_mask_type[:3]
        action_mask = (action_mask_loc, action_mask_type)
        return obs.replace(action_mask=action_mask)

    def get_obs(self, rng, state):
        obs = super().get_obs(rng, state)
        return self._reduce_action_mask(obs)
    
    def get_finished_obs(self, rng, state, not_done):
        obs = super().get_finished_obs(rng, state, not_done)
        return self._reduce_action_mask(obs)

TOTAL_OBJECT_TYPES = 5
class LocalKeyMazeEditor(LocalMazeEditor):

    @property
    def num_actions(self) -> int:
        return self.num_agents * self.agent_view_size**2 * TOTAL_OBJECT_TYPES
    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        dir_vec = DIR_TO_VEC[agent_dir]
        
        obs_fwd_bound1 = agent_pos
        obs_fwd_bound2 = agent_pos + dir_vec*(self.agent_view_size-1)

        side_offset = self.agent_view_size//2
        obs_side_bound1 = agent_pos + (dir_vec == 0)*side_offset
        obs_side_bound2 = agent_pos - (dir_vec == 0)*side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])
        padding = self.agent_view_size-1
        xmin, ymin = jnp.min(all_bounds, 0) + padding

        maze_map_w_cur_agent = maze_map.at[agent_pos[1] + self.agent_view_size-1, agent_pos[0] + self.agent_view_size-1].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['yellow'], agent_dir], dtype=jnp.uint8)
        )
        maze_map = jax.lax.select(include_agent, maze_map_w_cur_agent, maze_map)
        obs = jax.lax.dynamic_slice(maze_map, (ymin, xmin, 0), (self.agent_view_size, self.agent_view_size, 3))

        padded_observation_map = jnp.ones(maze_map.shape[0:2], dtype=jnp.bool_).at[padding:-padding, padding:-padding].set(level.observation_map)
        obs_map = jax.lax.dynamic_slice(padded_observation_map, (ymin, xmin), (self.agent_view_size, self.agent_view_size))

        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)] + [jnp.logical_and(~obs_map.flatten(), level.key_placed==0)] + [jnp.logical_and(~obs_map.flatten(), level.door_placed==0)])

        return obs, obs_map, action_mask

    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        edit_loc_idx = edit_idx % (self.agent_view_size**2)
        edit_type_idx = (edit_idx % (self.agent_view_size**2 * TOTAL_OBJECT_TYPES)) // self.agent_view_size**2
        edit_agent_idx = edit_idx // (self.agent_view_size**2 * TOTAL_OBJECT_TYPES)
        
        agent_pos = jax.lax.dynamic_index_in_dim(state.agent_locs, edit_agent_idx, axis=0).squeeze(0)
        agent_dir = jax.lax.dynamic_index_in_dim(state.agent_dirs, edit_agent_idx, axis=0).squeeze(0)

        dir_vec = DIR_TO_VEC[agent_dir]
        
        obs_fwd_bound1 = agent_pos
        obs_fwd_bound2 = agent_pos + dir_vec*(self.agent_view_size-1)

        side_offset = self.agent_view_size//2
        obs_side_bound1 = agent_pos + (dir_vec == 0)*side_offset
        obs_side_bound2 = agent_pos - (dir_vec == 0)*side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])
        xmin, ymin = jnp.min(all_bounds, 0)

        def move_goal():
            level = state.level
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin
            return jax.lax.cond(
                level.goal_placed,
                lambda: level,
                lambda : level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), goal_placed=jnp.array(True, dtype=jnp.bool_))
            )

        def move_key():
            level = state.level
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin
            return jax.lax.cond(
                level.key_placed,
                lambda: level,
                lambda : level.replace(wall_map=level.wall_map.at[y, x].set(False), key_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), key_placed=jnp.array(1, dtype=jnp.uint8))
            )

        def move_door():
            level = state.level
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin
            return jax.lax.cond(
                level.door_placed,
                lambda: level,
                lambda : level.replace(wall_map=level.wall_map.at[y, x].set(False), door_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True), door_placed=jnp.array(1, dtype=jnp.uint8))
            )
        
        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))
        
        def move_agent_x():
            level = state.level
            x = edit_idx % max_w
            y = level.agent_pos[1]
            return level.replace(agent_pos=jnp.array([x, y], dtype=jnp.uint32))
        
        def move_agent_y():
            level = state.level
            x = level.agent_pos[0]
            y = edit_idx % max_h
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=jnp.array([x, y], dtype=jnp.uint32), observation_map=level.observation_map.at[y, x].set(True))
        
        def toggle_wall():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin

            return level.replace(wall_map=level.wall_map.at[y, x].set(edit_type_idx != 0), observation_map=level.observation_map.at[y, x].set(True))
        
        def set_agent_locs():
            agent_locs=jnp.tile(level.agent_pos, (self.num_agents, 1))
            agent_dirs=jnp.tile(level.agent_dir, (self.num_agents))
            return state.replace(agent_locs=agent_locs, agent_dirs=agent_dirs)
        
        edit_goal = jnp.clip(edit_type_idx-1, 0, 3)
        edit_time = state.time.clip(None, 3)
        if self.set_start:
            level = jax.lax.switch(edit_time, [
                rotate_agent,
                move_agent_x,
                move_agent_y,
                lambda: jax.lax.switch(edit_goal, [toggle_wall, move_goal, move_key, move_door])
            ])

            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(edit_time <= 2, x, y),
                set_agent_locs(),
                state
            )
        else:
            level = jax.lax.switch(edit_goal, [toggle_wall, move_goal, move_key, move_door])
        
        return state.replace(level=level)

class LocalKeyMazeEditorRotate(LocalKeyMazeEditor):
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        def rotate(arr):
            return (agent_dir == 0)*jnp.rot90(arr, 1) + \
                   (agent_dir == 1)*jnp.rot90(arr, 2) + \
                   (agent_dir == 2)*jnp.rot90(arr, 3) + \
                   (agent_dir == 3)*jnp.rot90(arr, 4)
        
        obs, obs_map, _ = super().get_agent_obs(agent_pos, agent_dir, level, maze_map, include_agent)    
        obs, obs_map = jax.tree_util.tree_map(rotate, (obs, obs_map))
        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)] + [jnp.logical_and(~obs_map.flatten(), level.key_placed==0)] + [jnp.logical_and(~obs_map.flatten(), level.door_placed==0)])
        return obs, obs_map, action_mask
    
    def _edit_level(self, rng, state, edit_idx, params):
        edit_loc_idx = edit_idx % (self.agent_view_size**2)
        edit_agent_idx = edit_idx // (self.agent_view_size**2 * 3)
        
        agent_dir = jax.lax.dynamic_index_in_dim(state.agent_dirs, edit_agent_idx, axis=0).squeeze(0)

        def rot_index(idx):
            def rot_coords(x, y):
                x_r = (agent_dir == 0)*(self.agent_view_size - y - 1) + \
                    (agent_dir == 1)*(self.agent_view_size - x - 1) + \
                    (agent_dir == 2)*(y) + \
                    (agent_dir == 3)*(x)
                
                y_r = (agent_dir == 0)*(x) + \
                    (agent_dir == 1)*(self.agent_view_size - y - 1) + \
                    (agent_dir == 2)*(self.agent_view_size - x - 1) + \
                    (agent_dir == 3)*(y)
                return x_r, y_r
            
            x, y = idx % self.agent_view_size, idx // self.agent_view_size
            x_r, y_r = rot_coords(x, y)
            return x_r + y_r * self.agent_view_size
        
        r_edit_loc_idx = rot_index(edit_loc_idx)
        edit_idx = edit_idx + r_edit_loc_idx - edit_loc_idx
        return super()._edit_level(rng, state, edit_idx, params)

# Will only work for single agent - need to fix multi-agent action masking for action type selection
class LocalKeyMazeEditorRotateSplitAct(LocalKeyMazeEditorRotate):
    def action_space(self, params: EnvParams) -> MultiDiscrete:
        return MultiDiscrete(self.num_actions)

    @property
    def num_actions(self) -> int:
        return (self.num_agents * self.agent_view_size**2, TOTAL_OBJECT_TYPES)
    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        obs, obs_map, _ = super().get_agent_obs(agent_pos, agent_dir, level, maze_map, include_agent)    
        action_mask_loc = ~obs_map.flatten()
        action_mask_type = jnp.array([True, True, ~level.goal_placed, level.key_placed==0, level.door_placed==0])
        return obs, obs_map, (action_mask_loc, action_mask_type)
    
    def _edit_level(self, rng, state, edit_idxs, params):
        loc_idx, edit_type_idx = edit_idxs

        edit_loc_idx = loc_idx % (self.agent_view_size**2)
        edit_agent_idx = loc_idx // (self.agent_view_size**2)

        edit_idx = edit_loc_idx + (self.agent_view_size**2) * edit_type_idx + (self.agent_view_size**2 * TOTAL_OBJECT_TYPES) * edit_agent_idx
        
        return super()._edit_level(rng, state, edit_idx, params)

    def _reduce_action_mask(self, obs: Observation):
        action_mask_loc, action_mask_type = obs.action_mask
        action_mask_type = action_mask_type[:TOTAL_OBJECT_TYPES]
        action_mask = (action_mask_loc, action_mask_type)
        return obs.replace(action_mask=action_mask)

    def get_obs(self, rng, state):
        obs = super().get_obs(rng, state)
        return self._reduce_action_mask(obs)
    
    def get_finished_obs(self, rng, state, not_done):
        obs = super().get_finished_obs(rng, state, not_done)
        return self._reduce_action_mask(obs)

class BernoulliMazeEditor(UnderspecifiedEnv):
    """
        This environment allows the adversary to generate a level. The adversary can move the goal, move the agent, rotate the agent, or toggle walls.
        The action space is discrete, of dimension w*h, where w and h are the width and height of the maze, respectively. The first action moves the goal, the second action rotates the agent, the third action moves the agent, and the fourth action onwards toggles walls.
    """
    def __init__(self, env: Maze):
        super().__init__()
        self._env = env

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
        # Do not edit level if in terminal state
        rng, rng_obs = jax.random.split(rng)
        new_level = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(state.terminal, x, y),
            state.level,
            self._edit_level(rng, state, action, params)
        )
        # Check game condition & no. steps for termination condition
        state = state.replace(level=new_level, time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return self.get_obs(rng_obs, state), state, 0, done, {}
        
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = self.init_state_from_level(level)
        return self.get_obs(rng, state), state
    

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, self.num_actions)
    
    # ===

    @property
    def num_actions(self) -> int:
        return self._env.max_width * self._env.max_height

    def get_obs(self, rng: chex.Array, state: EnvState):
        goal_idx = state.level.goal_pos[1] * self._env.max_width + state.level.goal_pos[0]
        agent_idx = state.level.agent_pos[1] * self._env.max_width + state.level.agent_pos[0]
        
        #action_mask = jnp.concatenate([~state.level.observation_map.flatten()]*2)
        
        maze_map = make_maze_map(state.level)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1], state.level.agent_pos[0]].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
    
        return Observation(
            image=maze_map_with_agent,
            observation_map = None,
            action_mask=None,
            agent_observations = None,
            agent_steps = None,
            time=state.time,
            random_z = None,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return False
        
    def init_state_from_level(self, level):
        return EnvState(
            level=level,
            time=jnp.array(0, dtype=jnp.uint32),
            terminal=False,
            agent_locs=None,
            agent_dirs=None
        )
        
    def _edit_level(self, rng: chex.PRNGKey, state: EnvState, edit_array: chex.Array, params: EnvParams) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height
        level = state.level

        goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
        agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]

        wall_updates = (edit_array.at[goal_idx].set(False)).at[agent_idx].set(False).reshape(max_h, max_w)
        level = level.replace(wall_map=jnp.logical_xor(level.wall_map, wall_updates))

        def move_goal():
            p = ~level.wall_map.flatten().at[agent_idx].set(True)
            new_goal_idx = jax.random.choice(rng, max_w * max_h, p=p),
            x, y = new_goal_idx % max_w, new_goal_idx // max_w
            return level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=jnp.array([x, y], dtype=jnp.uint32), goal_placed=jnp.array(True, dtype=jnp.bool_))
        
        #level = jax.lax.switch(edit_array.at[goal_idx].astype(jnp.uint8), [move_goal, lambda : level])
        
        return level