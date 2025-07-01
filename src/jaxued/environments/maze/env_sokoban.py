from typing import Tuple
from enum import IntEnum
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from jaxued.environments import UnderspecifiedEnv
from .level import Level, prefabs
from .env import Actions, EnvState, Observation, EnvParams, Maze, make_maze_map
    
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
    pass