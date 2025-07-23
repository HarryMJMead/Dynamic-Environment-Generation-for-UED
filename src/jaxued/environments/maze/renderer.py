import numpy as np
import jax
import jax.numpy as jnp
import chex

from jaxued.environments.underspecified_env import EnvParams, EnvState
from .env import DIR_TO_VEC, Maze
from functools import partial

class MazeRenderer(object):
    """This class renders the maze for visual logging, compatible with jit.

        Args:
            env (Maze): 
            tile_size (int, optional): The number of pixels each tile should take up. Defaults to 32.
            render_border (bool, optional): If true, renders the one-tile thick border around the level. Defaults to True.
    """
    def __init__(self, env: Maze, tile_size: int=32, render_border: bool=True, render_boxes: bool=False):
        self.env = env
        self.tile_size = tile_size
        self.render_border = render_border
        self._atlas = jnp.array(_make_tile_atlas(tile_size))
        self._show_observations = False
        self._render_boxes = render_boxes
        
    @partial(jax.jit, static_argnums=(0,))
    def render_level(self, level, env_params):
        # For Minigrid, env_state contains all attributes of level,
        # and only uses these attributes. So can just call render_state.
        # However, in general, these routines may be a bit different.
        # For example, levels may map to many different start states.
        # As such, one may want to render an image representative of all
        # possible start states when rendering a level.
        return self.render_state(level, env_params)
    
    @partial(jax.jit, static_argnums=(0,))
    def render_state(self, env_state: EnvState, env_params: EnvParams) -> chex.Array:
        tile_size = self.tile_size
        nrows = self.env.max_height + 2*self.render_border
        ncols = self.env.max_width + 2*self.render_border
        width_px = ncols * tile_size
        height_px = nrows * tile_size
        
        agent_pos = env_state.agent_pos + self.render_border
        goal_pos = env_state.goal_pos + self.render_border
        door_pos = env_state.door_pos + self.render_border
        key_pos = env_state.key_pos + self.render_border
        
        cells = jnp.where(env_state.wall_map, 1, 0)
        if self._render_boxes:
            cells = jnp.where(~env_state.box_map, cells, jnp.ones_like(cells)*12)
        if self._show_observations:
            cells = jnp.where(env_state.observation_map, cells, jnp.ones_like(cells)*7)
        if self.render_border:
            cells = jnp.pad(cells, 1, mode="constant", constant_values=True)

        cells = cells.at[agent_pos[1], agent_pos[0]].set(3 + env_state.agent_dir)
        cells = jax.lax.cond(
            env_state.goal_placed,
            lambda : cells.at[goal_pos[1], goal_pos[0]].set(2),
            lambda : cells
        )
        cells = jax.lax.cond(
            env_state.door_placed == 1,
            lambda : cells.at[door_pos[1], door_pos[0]].set(8),
            lambda : cells
        )
        cells = jax.lax.cond(
            env_state.key_placed == 1,
            lambda : cells.at[key_pos[1], key_pos[0]].set(9),
            lambda : cells
        )
        cells = jax.lax.cond(
            env_state.door_placed == 2,
            lambda : cells.at[door_pos[1], door_pos[0]].set(10),
            lambda : cells
        )
        cells = jax.lax.cond(
            env_state.key_placed == 2,
            lambda : cells.at[key_pos[1], key_pos[0]].set(11),
            lambda : cells
        )

        img = self._atlas[cells].transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)
        
        f_vec = DIR_TO_VEC[env_state.agent_dir]
        r_vec = jnp.array([-f_vec[1], f_vec[0]])

        agent_view_size = self.env.agent_view_size

        min_bound = jnp.min(jnp.stack([
            agent_pos, 
            agent_pos + f_vec*(agent_view_size-1), 
            agent_pos - r_vec*(agent_view_size//2), 
            agent_pos + r_vec*(agent_view_size//2),
        ]), 0)

        min_x = jnp.minimum(jnp.maximum(min_bound[0], 0), env_state.wall_map.shape[0] - 1 + 2*self.render_border)
        min_y = jnp.minimum(jnp.maximum(min_bound[1], 0), env_state.wall_map.shape[1] - 1 + 2*self.render_border)
        max_x = jnp.minimum(jnp.maximum(min_bound[0]+agent_view_size, 0), env_state.wall_map.shape[0] + 2*self.render_border)
        max_y = jnp.minimum(jnp.maximum(min_bound[1]+agent_view_size, 0), env_state.wall_map.shape[1] + 2*self.render_border)
        
        all_pos = jnp.arange(ncols * nrows)
        mask = \
            ((all_pos % ncols) >= min_x) & \
            ((all_pos % ncols) < max_x) & \
            ((all_pos // ncols) >= min_y) & \
            ((all_pos // ncols) < max_y)
        mask = jnp.kron(mask.reshape(nrows, ncols), jnp.ones((self.tile_size, self.tile_size)))[..., None]
        
        highlight_img = (img + 0.3 * (255 - img)).astype(jnp.uint8).clip(0, 255)
        return jnp.where(mask, highlight_img, img)

# This function used to have more differences than just show observations - kept like this to not break old code requiring ObserveMmazeRender
class ObservedMazeRenderer(MazeRenderer):
    def __init__(self, env: Maze, tile_size: int=32, render_border: bool=True, render_boxes: bool=True):
        super().__init__(env, tile_size, render_border, render_boxes)
        self._show_observations = True

class LocalObservedMazeRenderer(ObservedMazeRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render_state(self, env_state: EnvState, env_params: EnvParams) -> chex.Array:
        tile_size = self.tile_size
        nrows = self.env.max_height + 2*self.render_border
        ncols = self.env.max_width + 2*self.render_border
        width_px = ncols * tile_size
        height_px = nrows * tile_size

        image = super(self.__class__, self).render_state(env_state.level, env_params)

        agents = jnp.zeros(env_state.level.wall_map.shape, dtype=jnp.uint8)
        agents = agents.at[env_state.agent_locs[:, 1], env_state.agent_locs[:, 0]].set(3 + env_state.agent_dirs)

        color_mask = jnp.zeros(env_state.level.wall_map.shape, dtype=jnp.uint8)
        color_mask = color_mask.at[env_state.agent_locs[:, 1], env_state.agent_locs[:, 0]].set(jnp.linspace(100, 255, env_state.agent_dirs.shape[0]).astype(jnp.uint8))
        
        if self._render_boxes:
            agents = jnp.where(env_state.box_locs.any(axis=0), 12, agents)
            box_color_mask = (env_state.box_locs * jnp.linspace(100, 255, 2)[..., None, None]).max(axis=0).astype(jnp.uint8)
            color_mask = jnp.where(env_state.box_locs.any(axis=0), box_color_mask, color_mask)

        if self.render_border:
            color_mask = jnp.pad(color_mask, 1, mode="constant", constant_values=True)
            agents = jnp.pad(agents, 1, mode="constant", constant_values=True)

        color_image = jnp.kron(color_mask, jnp.ones((tile_size, tile_size), dtype=color_mask.dtype))
        agent_image = self._atlas[agents].transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)

        agent_color_image = agent_image.at[:, :, 2].add(color_image).clip(0, 255)

        return jnp.where(color_image[..., None], agent_color_image, image)


def _make_tile_atlas(tile_size):
    TRI_COORDS = np.array([
        [0.12, 0.19],
        [0.87, 0.50],
        [0.12, 0.81],
    ])
    
    def fill_coords(img, fn, color):
        new_img = img.copy()
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                yf = (y + 0.5) / img.shape[0]
                xf = (x + 0.5) / img.shape[1]
                if fn(xf, yf):
                    new_img[y, x] = color
        return new_img

    def point_in_rect(xmin, xmax, ymin, ymax):
        def fn(x, y):
            return x >= xmin and x <= xmax and y >= ymin and y <= ymax
        return fn

    def point_in_triangle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        def fn(x, y):
            v0 = c - a
            v1 = b - a
            v2 = np.array((x, y)) - a

            # Compute dot products
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            # Compute barycentric coordinates
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            # Check if point is in triangle
            return (u >= 0) and (v >= 0) and (u + v) < 1

        return fn
    
    atlas = np.empty((13, tile_size, tile_size, 3), dtype=np.uint8)
    
    def add_border(tile):
        new_tile = fill_coords(tile, point_in_rect(0, 0.031, 0, 1), (100, 100, 100)) 
        return fill_coords(new_tile, point_in_rect(0, 1, 0, 0.031), (100, 100, 100)) 
    
    atlas[0] = add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1))) # empty
    atlas[1] = np.tile([100, 100, 100], (tile_size, tile_size, 1)) # wall
    atlas[2] = np.tile([0, 255, 0], (tile_size, tile_size, 1)) # goal
    
    # Handle player
    agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    agent_tile = fill_coords(agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])
    
    atlas[3] = add_border(agent_tile) # right
    atlas[4] = add_border(np.rot90(agent_tile, k=3)) # down
    atlas[5] = add_border(np.rot90(agent_tile, k=2)) # left
    atlas[6] = add_border(np.rot90(agent_tile, k=1)) # up

    # Observation Tracking
    atlas[7] = np.tile([100, 0, 100], (tile_size, tile_size, 1)) # unobserved

    # Door
    BORDER_WIDTH = 0.1
    door_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    door_tile = fill_coords(door_tile, point_in_rect(0, BORDER_WIDTH, 0, 1), (0, 0, 255))
    door_tile = fill_coords(door_tile, point_in_rect(1-BORDER_WIDTH, 1, 0, 1), (0, 0, 255))
    door_tile = fill_coords(door_tile, point_in_rect(0, 1, 0, BORDER_WIDTH), (0, 0, 255))
    door_tile = fill_coords(door_tile, point_in_rect(0, 1, 1-BORDER_WIDTH, 1), (0, 0, 255))
    atlas[8] = door_tile

    # Key
    KEY_TRI_COORDS = np.array([
        [0.1, 0.1],
        [0.1, 0.50],
        [0.9, 0.1],
    ])
    key_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    key_tile = fill_coords(key_tile, point_in_triangle(*KEY_TRI_COORDS), [0, 0, 255])
    key_tile = fill_coords(key_tile, point_in_triangle(*(KEY_TRI_COORDS+jnp.array([0, 0.4]))), [0, 0, 255])
    atlas[9] = key_tile

    # Door Open
    BORDER_WIDTH = 0.1
    door_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    door_tile = fill_coords(door_tile, point_in_rect(0, BORDER_WIDTH, 0, 1), (0, 255, 255))
    door_tile = fill_coords(door_tile, point_in_rect(1-BORDER_WIDTH, 1, 0, 1), (0, 255, 255))
    door_tile = fill_coords(door_tile, point_in_rect(0, 1, 0, BORDER_WIDTH), (0, 255, 255))
    door_tile = fill_coords(door_tile, point_in_rect(0, 1, 1-BORDER_WIDTH, 1), (0, 255, 255))
    atlas[10] = door_tile

    # Key Picked Up
    KEY_TRI_COORDS = np.array([
        [0.1, 0.1],
        [0.1, 0.50],
        [0.9, 0.1],
    ])
    key_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    key_tile = fill_coords(key_tile, point_in_triangle(*KEY_TRI_COORDS), [0, 255, 255])
    key_tile = fill_coords(key_tile, point_in_triangle(*(KEY_TRI_COORDS+jnp.array([0, 0.4]))), [0, 255, 255])
    atlas[11] = key_tile

    # Box
    BORDER_WIDTH = 0.1
    box_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    box_tile = fill_coords(box_tile, point_in_rect(0, BORDER_WIDTH, 0, 1), (255, 255, 0))
    box_tile = fill_coords(box_tile, point_in_rect(1-BORDER_WIDTH, 1, 0, 1), (255, 255, 0))
    box_tile = fill_coords(box_tile, point_in_rect(0, 1, 0, BORDER_WIDTH), (255, 255, 0))
    box_tile = fill_coords(box_tile, point_in_rect(0, 1, 1-BORDER_WIDTH, 1), (255, 255, 0))

    CENTRE_WIDTH = 0.15
    box_tile = fill_coords(box_tile, point_in_rect(0.5-CENTRE_WIDTH/2, 0.5+CENTRE_WIDTH/2, 0, 1), (255, 255, 0))
    box_tile = fill_coords(box_tile, point_in_rect(0, 1, 0.5-CENTRE_WIDTH/2, 0.5+CENTRE_WIDTH/2), (255, 255, 0))
    atlas[12] = box_tile

    return atlas