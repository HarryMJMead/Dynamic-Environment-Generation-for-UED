from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from .level import Level, ObservedLevel
from jaxued.environments import UnderspecifiedEnv
from .env import COLOR_TO_INDEX, OBJECT_TO_INDEX, Maze, make_maze_map
from .env_editor import LocalMazeEditor, EnvState, Observation, EnvParams, MultiDiscrete

DIR_TO_VEC = jnp.array([
    (1, 0), # right
    (0, 1), # down
    (-1, 0), # left
    (0, -1), # up
], dtype=jnp.int8)

TOTAL_OBJECT_TYPES = 4
class LocalSokobanMazeEditor(LocalMazeEditor):
    def __init__(self, env: Maze, random_z_dimensions: int = 16, zero_out_random_z: bool = False, num_agents = 2, agent_view_size = 5, set_start = False, set_init_pos = True):
        super().__init__(
            env, 
            random_z_dimensions, 
            zero_out_random_z, 
            num_agents, 
            agent_view_size, 
            set_start
        )
        self.show_boxes = True

    @property
    def num_actions(self) -> int:
        return self.num_agents * self.agent_view_size**2 * TOTAL_OBJECT_TYPES
    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, box_map: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
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

        padded_observation_map = jnp.pad(
            level.observation_map,
            pad_width=((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=True
        )
        obs_map = jax.lax.dynamic_slice(padded_observation_map, (ymin, xmin), (self.agent_view_size, self.agent_view_size))

        padded_box_map = jnp.pad(
            box_map,
            pad_width=((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=False
        )
        box_map = jax.lax.dynamic_slice(padded_box_map, (ymin, xmin), (self.agent_view_size, self.agent_view_size))

        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)] + [jnp.logical_and(~obs_map.flatten(), True)])

        return obs, obs_map, box_map, action_mask

    def get_finished_obs(self, rng: chex.Array, state: EnvState, not_done: chex.Array):
        maze_map = make_maze_map(state.level, padding=self.agent_view_size-1, show_boxes=self.show_boxes)
        maze_map_with_agent = maze_map.at[state.level.agent_pos[1] + self.agent_view_size-1, state.level.agent_pos[0] + self.agent_view_size-1].set(
            jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], state.level.agent_dir], dtype=jnp.uint8)
        )
        image, obs_map, box_map, action_mask = jax.vmap(self.get_agent_obs, in_axes=(0, 0, 0, None, None))(state.agent_locs, state.agent_dirs, state.box_locs, state.level, maze_map_with_agent)
        finished_obs, _, _, _ = self.get_agent_obs(state.level.goal_pos - jnp.array([2, 0]), jnp.array(0), state.level.box_map, state.level, maze_map_with_agent, False)
        finished_image = finished_obs[None, ...].repeat(self.num_agents, axis=0)
        image = jnp.where(not_done.reshape(-1, 1, 1, 1), image, finished_image)
        image = jnp.moveaxis(image, 0, -2).reshape(self.agent_view_size, self.agent_view_size, 3*self.num_agents)
        obs_map = jnp.moveaxis(obs_map, 0, -1)
        box_map = jnp.moveaxis(box_map, 0, -1)
        action_mask = jax.tree_map(lambda x: x.flatten(), action_mask)

        if self.set_start:
            action_mask = jax.lax.select(state.time <= 2, self.agent_placement_action_mask(state.time), action_mask)
    
        return Observation(
            image=image/10,
            observation_map=obs_map,
            action_mask=action_mask,
            agent_observations=None,
            agent_steps=jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
            place_goal=jnp.array(True, dtype=jnp.bool_),
            goal_placed=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            key_placed=jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            door_placed=jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            agent_dirs=state.agent_dirs,
            agent_values=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            agent_boxes=box_map,
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
        image, obs_map, box_map, action_mask = jax.vmap(self.get_agent_obs, in_axes=(0, 0, 0, None, None))(state.agent_locs, state.agent_dirs, state.box_locs, state.level, maze_map_with_agent)
        image = jnp.moveaxis(image, 0, -2).reshape(self.agent_view_size, self.agent_view_size, 3*self.num_agents)
        obs_map = jnp.moveaxis(obs_map, 0, -1)
        box_map = jnp.moveaxis(box_map, 0, -1)
        action_mask = jax.tree_map(lambda x: x.flatten(), action_mask)

        if self.set_start:
            action_mask = jax.lax.select(state.time <= 2, self.agent_placement_action_mask(state.time), action_mask)
    
        return Observation(
            image=image/10,
            observation_map=obs_map,
            action_mask=action_mask,
            agent_observations=None,
            agent_steps=jnp.array(0, dtype=jnp.int32),
            time=state.time,
            random_z=(jax.random.uniform(rng, (self.random_z_dimensions,)) * self.zero_out_random_z).astype(jnp.float32),
            place_goal=jnp.array(True, dtype=jnp.bool_),
            goal_placed=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            key_placed=jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            door_placed=jnp.zeros((self.num_agents,), dtype=jnp.uint8),
            agent_dirs=state.agent_dirs,
            agent_values=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            agent_boxes=box_map,
        )

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
        
        def toggle_box():
            level = state.level
            
            # if attempting to toggle wall on top of agent or goal, do nothing
            x, y = edit_loc_idx % self.agent_view_size + xmin, edit_loc_idx // self.agent_view_size + ymin

            return level.replace(box_map=level.box_map.at[y, x].set(True), observation_map=level.observation_map.at[y, x].set(True))
        
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
                lambda: jax.lax.switch(edit_goal, [toggle_wall, move_goal, toggle_box])
            ])

            state = jax.tree_map(
                lambda x, y: jax.lax.select(edit_time <= 2, x, y),
                set_agent_locs(),
                state
            )
        else:
            level = jax.lax.switch(edit_goal, [toggle_wall, move_goal, toggle_box])
        
        return state.replace(level=level)

class LocalSokobanMazeEditorRotate(LocalSokobanMazeEditor):
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, box_map: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        def rotate(arr):
            return (agent_dir == 0)*jnp.rot90(arr, 1) + \
                   (agent_dir == 1)*jnp.rot90(arr, 2) + \
                   (agent_dir == 2)*jnp.rot90(arr, 3) + \
                   (agent_dir == 3)*jnp.rot90(arr, 4)
        
        obs, obs_map, box_map, _ = super().get_agent_obs(agent_pos, agent_dir, box_map, level, maze_map, include_agent)    
        obs, obs_map, box_map = jax.tree_map(rotate, (obs, obs_map, box_map))
        action_mask = jnp.concatenate([~obs_map.flatten()]*2 + [jnp.logical_and(~obs_map.flatten(), ~level.goal_placed)] + [jnp.logical_and(~obs_map.flatten(), True)])
        return obs, obs_map, box_map, action_mask
    
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
class LocalSokobanMazeEditorRotateSplitAct(LocalSokobanMazeEditorRotate):
    def action_space(self, params: EnvParams) -> MultiDiscrete:
        return MultiDiscrete(self.num_actions)

    @property
    def num_actions(self) -> int:
        return (self.num_agents * self.agent_view_size**2, TOTAL_OBJECT_TYPES)
    
    def get_agent_obs(self, agent_pos: chex.Array, agent_dir: chex.Array, box_map: chex.Array, level: ObservedLevel, maze_map: chex.Array, include_agent=True):
        obs, obs_map, box_map, _ = super().get_agent_obs(agent_pos, agent_dir, box_map, level, maze_map, include_agent)    
        action_mask_loc = ~obs_map.flatten()
        action_mask_type = jnp.array([True, True, ~level.goal_placed, True])
        return obs, obs_map, box_map, (action_mask_loc, action_mask_type)
    
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