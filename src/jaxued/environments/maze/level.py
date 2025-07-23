import numpy as np
import chex
from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Level:
    """This represents a level in the maze environment. The main features are the wall map, goal position, agent position and agent direction.
    """
    wall_map: chex.Array
    goal_pos: chex.Array
    agent_pos: chex.Array
    agent_dir: int
    width: int
    height: int
    goal_placed: chex.Array

    key_pos: chex.Array = jnp.array([0, 0], dtype=jnp.uint32)
    key_placed: chex.Array = jnp.array(0, dtype=jnp.uint8)
    door_pos: chex.Array = jnp.array([0, 0], dtype=jnp.uint32)
    door_placed: chex.Array = jnp.array(0, dtype=jnp.uint8)
    
    box_map: chex.Array = jnp.array([[0]], dtype=jnp.bool_)
    
    def is_well_formatted(self):
        wall_map_is_binary = jnp.all((self.wall_map == 0) | (self.wall_map == 1))
        agent_goal_pos_distinct = ~(jnp.all(self.agent_pos == self.goal_pos))
        agent_dir_valid = (0 <= self.agent_dir) & (self.agent_dir <= 4)
        agent_not_on_wall = ~(self.wall_map[self.agent_pos[1], self.agent_pos[0]])
        goal_not_on_wall = ~(self.wall_map[self.goal_pos[1], self.goal_pos[0]])
        agent_within_bounds = (0 <= self.agent_pos[0]) & (self.agent_pos[0] < self.width) & (0 <= self.agent_pos[1]) & (self.agent_pos[1] < self.height)
        goal_within_bounds = (0 <= self.goal_pos[0]) & (self.goal_pos[0] < self.width) & (0 <= self.goal_pos[1]) & (self.goal_pos[1] < self.height)
        well_formatted = wall_map_is_binary & agent_goal_pos_distinct & agent_dir_valid & agent_not_on_wall & goal_not_on_wall & agent_within_bounds & goal_within_bounds
        return well_formatted
    
    @classmethod
    def from_str(cls, level_str):
        level_str = level_str.strip()
        rows = level_str.split('\n')
        nrows = len(rows)
        assert all(len(row) == len(rows[0]) for row in rows), "All rows must have same length"
        ncols = len(rows[0])
        
        wall_map = np.zeros((nrows, ncols), dtype=bool)
        goal_pos = []
        agent_pos = None
        agent_dir = None

        key_pos = [(0, 0)]
        door_pos = [(0, 0)]
        box_map = np.zeros((nrows, ncols), dtype=bool)  
        
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c == '#':
                    wall_map[y, x] = True
                elif c == 'B':
                    box_map[y, x] = True
                elif c == 'G':
                    goal_pos.append((x, y))
                elif c == 'K':
                    key_pos.append((x, y))
                elif c == 'D':
                    door_pos.append((x, y))
                elif c == '>':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 0
                elif c == 'v':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 1
                elif c == '<':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 2
                elif c == '^':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 3
                elif c == '.':
                    pass
                else:
                    raise Exception("Unexpected character.")
        
        assert len(goal_pos) > 0, "Goal position not set."
        assert agent_pos is not None, "Agent position not set."
        
        return Level(jnp.array(wall_map), *map(lambda x: jnp.array(x, dtype=jnp.uint32), (goal_pos[0], agent_pos)), jnp.array(agent_dir, dtype=jnp.uint8), ncols, nrows, jnp.array(True, dtype=jnp.bool_), 
                     door_pos=jnp.array(door_pos[-1], dtype=jnp.uint32), key_pos=jnp.array(key_pos[-1], dtype=jnp.uint32), door_placed=jnp.array(len(door_pos)>1, dtype=jnp.uint8), key_placed=jnp.array(len(key_pos)>1, dtype=jnp.uint8), box_map=box_map)
    
    def to_str(self):
        w, h = self.width, self.height
        h, w = self.wall_map.shape
        enc = np.full((h, w), None)
    
        for y in range(h):
            for x in range(w):
                enc[y, x] = '#' if self.wall_map[y, x] else '.'
        
        x, y = self.agent_pos
        if self.agent_dir == 0:
            agent_char = '>'
        elif self.agent_dir == 1:
            agent_char = 'v'
        elif self.agent_dir == 2:
            agent_char = '<'
        elif self.agent_dir == 3:
            agent_char = '^'    
        enc[y, x] = agent_char
        
        p = self.goal_pos
        x, y = self.goal_pos
        enc[y, x] = 'G'

        x, y = self.key_pos
        if self.key_placed:
            enc[y, x] = 'K'

        x, y = self.door_pos
        if self.door_placed:
            enc[y, x] = 'Ds'
        
        return '\n'.join([''.join(row) for row in enc]).strip()
    
    def pad_to_shape(self, max_width, max_height):
        batch_dims = self.wall_map.shape[:-2]
        h, w = self.wall_map.shape[-2:]
        assert max_width >= w and max_height >= h  
        new_wall_map = jax.lax.dynamic_update_slice(
            jnp.ones((*batch_dims, max_height, max_width), dtype=jnp.bool_),
            self.wall_map,
            (*(0,)*len(batch_dims), 0, 0),
        )
        return self.replace(wall_map=new_wall_map)
    
    @classmethod
    def stack(cls, levels):
        level_dims = np.array([[level.wall_map.shape[1], level.wall_map.shape[0]] for level in levels])
        max_width, max_height = level_dims.max(axis=0)
        return jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *(level.pad_to_shape(max_width, max_height) for level in levels)
        )
    
    @classmethod
    def load_prefabs(cls, ids):
        return Level.stack([Level.from_str(prefabs[id]) for id in ids])

TrivialMaze = """
...
.#.
>#G
"""

TrivialMaze2 = """
.....
.....
..#..
..#..
>.#.G
"""

TrivialMaze3 = """
.......
.......
.......
...#...
...#...
...#...
>..#..G
"""
        
SixteenRooms = """
...#..#..#...
.>.......#...
...#..#......
#.###.##.###.
...#.........
......#..#...
##.#.##.###.#
...#.....#...
...#..#......
.####.##.#.##
...#..#..#...
......#....G.
...#.....#...
"""

SixteenRooms2 = """
...#.....#...
.>....#..#...
...#..#..#...
####.##.###.#
...#..#......
......#..#...
#.#####.#####
...#..#..#...
...#.........
##.##.##.####
...#..#..#...
......#....G.
...#..#..#...
"""

Labyrinth = """
.............
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#...#...#.#.
.#########.#.
.....#.....#.
####.#.#####.
>....#.......
"""

LabyrinthFlipped = """
.............
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#.#...#...#.
.#.#########.
.#.....#.....
.#####.#.####
.......#....<
"""

Labyrinth2 = """
>#...........
.#.#########.
.#.#.......#.
.#.#.#####.#.
.#.#.#...#.#.
...#.#.#.#.#.
####.#G#.#.#.
...#.###.#.#.
.#.#.....#.#.
.#.#######.#.
.#.........#.
.###########.
.............
"""

StandardMaze = """
.....#>...#..
.###.####.##.
.#...........
.########.###
........#....
######.#####.
....#..#.....
.##...##.####
..#.#..#...#.
#.#.##.###.#.
#.#..#...#...
#.##.###.###.
...#..G#.#...
"""

StandardMaze2 = """
...#.#....#..
.#.#.####...#
.#........#..
.########.###
...#..#.#.#.G
##.#.##.#.#..
>#.#....#.##.
.#.##.###..#.
.#..#..###.#.
.##.##.#.#.#.
.#...#.#.#.#.
.#.#.#.#.#.#.
...#...#.....
"""

StandardMaze3 = """
...>#.#......
.####.#.####.
.#....#.#....
...####.#.#.#
##.#....#.#..
...#.##.#.##.
.#.#.#..#..#G
.#.#.#.###.##
.#...#.#.#...
.###.#.#.###.
.#...#.#...#.
.#.###.#.#.#.
.#...#...#...
"""

SixteenRooms_Key = """
...#..#..#..K
.>.......#...
...#..#......
#.###.##.###.
...#.........
......#..#...
##.#.##.###.#
...#.....#...
...#..#......
.####.##.####
...#..#..#...
......#..D.G.
...#.....#...
"""

SixteenRooms2_Key = """
...#.....#...
.>....#..#...
...#..#..#...
####.##.###.#
...#..#......
......#..#...
#.#####.#####
...#..#..#...
...#.........
##.##.##.####
...#..#..#...
......#..D.G.
K..#..#..#...
"""

Labyrinth_Key = """
K............
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#...#...#.#.
.#########.#.
.....#.....#.
####.#D#####.
>....#.......
"""

LabyrinthFlipped_Key = """
............K
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#.#...#...#.
.#.#########.
.#.....#.....
.#####D#.####
.......#....<
"""

Labyrinth2_Key = """
>#..........K
.#.#########.
.#.#.......#.
.#.#.#####.#.
.#.#.#...#.#.
...#.#.#.#.#.
####.#G#.#.#.
...#.###.#.#.
.#.#D....#.#.
.#.#######.#.
.#.........#.
.###########.
.............
"""

StandardMaze_Key = """
.....#>...#..
.###D####.##.
.#...........
.########.###
........#....
######.#####.
....#..#.....
.##...##.####
..#.#..#...#K
#.#.##.###.#.
#.#..#...#...
#.##.###.###.
...#..G#.#...
"""

StandardMaze2_Key = """
...#.#....#..
.#.#.####...#
.#........#..
.########D###
...#..#K#.#.G
##.#.##.#.#..
>#.#....#.##.
.#.##.###..#.
.#..#..###.#.
.##.##.#.#.#.
.#...#.#.#.#.
.#.#.#.#.#.#.
...#...#.....
"""

StandardMaze3_Key = """
...>#.#......
.####.#.####.
.#....#.#....
...####.#.#D#
##.#....#.#..
...#.##.#.##.
.#.#.#..#..#G
.#.#.#.###.##
.#...#.#.#...
.###.#.#.###.
.#...#.#...#.
.#.###.#.#.#.
.#..K#...#...
"""

FourRooms_Key = """
......#......
.>....#....G.
......#......
......#......
......#......
......#......
###.#####D###
......#......
......#......
......#......
.............
......#......
K.....#......
"""

FourRooms2_Key = """
......#......
.>....#....G.
......D......
......#......
......#......
......#......
###.#########
......#.....K
......#......
......#......
.............
......#......
......#......
"""

FourRooms3_Key = """
......#..>...
......#......
......#......
.............
......#......
......#......
###D#####.###
......#......
......#......
......#......
......#......
.G....#......
......#.....K
"""

SixteenRooms_Sokoban = """
...#..#..#...
.>.......#...
...#..#......
#.###.##.###.
...#.........
......#..#...
##.#.##.###B#
...#.....#...
...#..#...B..
.####B##B#.##
...#..#..#...
..B...#..#.G.
...#.........
"""

SixteenRooms2_Sokoban = """
...#.....#...
.>....#B.#...
...#..#..#...
####.##.###.#
...#..#......
.B....#..#...
#.#####.#####
...#..#..#...
.#.#...B.....
##.##.##B####
...#..#..#...
......#..#.G.
...#..#....#.
"""

FourRooms_Sokoban = """
......#......
.>....#....G.
......#......
......#......
...B..#..#...
......#......
###.#####.###
...B..#..B...
......#......
......#......
.....BB......
......#......
......#......
"""

FourRooms2_Sokoban = """
......#......
.>....#....G.
.....B.B.....
......#......
......#......
......#......
###B#########
......#......
......#......
......#......
......B......
......#......
......#......
"""

FourRooms3_Sokoban = """
...#..#..>...
...#..#......
...#..#......
.#....B......
..B#..#......
....#.#......
###.#####B###
......#......
......#......
......#......
......#......
.G....#......
......#......
"""

Detour_Sokoban = """
......#....#.
......#G.....
..###.#####.#
....#.#.#..B.
..#.#.#.#.#.#
..#.#.#.#.#..
..#.#.^.#.#..
..#.#.#B#.#G.
....#.#.#..B.
..###.#.###..
......#......
......#......
......#......
"""

Detour2_Sokoban = """
......#......
.B#...#......
..###.#.###..
.B..#.#.#....
..#.#.#.#.#..
..#G#.#.#.#..
..#.#.^.#.#..
..#.#.#.#.#..
.B..#.#.#....
..###.#.###..
.B#...#......
......#......
......#......
"""

MultiStep_Sokoban = """
......#......
.>....#....G.
......#......
......#......
......#......
....BB.#.....
.....BB......
.....B.#.....
......#......
......#......
......#......
......#......
......#......
"""


prefabs = {
    "TrivialMaze": TrivialMaze.strip(),
    "TrivialMaze2": TrivialMaze2.strip(),
    "TrivialMaze3": TrivialMaze3.strip(),
    "SixteenRooms": SixteenRooms.strip(),
    "SixteenRooms2": SixteenRooms2.strip(),
    "Labyrinth": Labyrinth.strip(),
    "LabyrinthFlipped": LabyrinthFlipped.strip(),
    "Labyrinth2": Labyrinth2.strip(),
    "StandardMaze": StandardMaze.strip(),
    "StandardMaze2": StandardMaze2.strip(),
    "StandardMaze3": StandardMaze3.strip(),

    "SixteenRooms_Key": SixteenRooms_Key.strip(),
    "SixteenRooms2_Key": SixteenRooms2_Key.strip(),
    "Labyrinth_Key": Labyrinth_Key.strip(),
    "LabyrinthFlipped_Key": LabyrinthFlipped_Key.strip(),
    "Labyrinth2_Key": Labyrinth2_Key.strip(),
    "StandardMaze_Key": StandardMaze_Key.strip(),
    "StandardMaze2_Key": StandardMaze2_Key.strip(),
    "StandardMaze3_Key": StandardMaze3_Key.strip(),
    "FourRooms_Key": FourRooms_Key.strip(),
    "FourRooms2_Key": FourRooms2_Key.strip(),
    "FourRooms3_Key": FourRooms3_Key.strip(),

    "SixteenRooms_Sokoban": SixteenRooms_Sokoban.strip(),
    "SixteenRooms2_Sokoban": SixteenRooms2_Sokoban.strip(),
    "FourRooms_Sokoban": FourRooms_Sokoban.strip(),
    "FourRooms2_Sokoban": FourRooms2_Sokoban.strip(),
    "FourRooms3_Sokoban": FourRooms3_Sokoban.strip(),
    "Detour_Sokoban": Detour_Sokoban.strip(),
    "Detour2_Sokoban": Detour2_Sokoban.strip(),
    "MultiStep_Sokoban": MultiStep_Sokoban.strip(),
}

@struct.dataclass
class ObservedLevel(Level):
    observation_map: chex.Array = None

    @classmethod
    def from_str(cls, level_str):
        level = Level.from_str(level_str)
        # By default, observation_map is fully observed (all True)
        observation_map = jnp.ones_like(level.wall_map, dtype=jnp.bool_)
        return ObservedLevel(
            wall_map=level.wall_map,
            goal_pos=level.goal_pos,
            agent_pos=level.agent_pos,
            agent_dir=level.agent_dir,
            width=level.width,
            height=level.height,
            goal_placed=level.goal_placed,
            key_pos=level.key_pos,
            key_placed=level.key_placed,
            door_pos=level.door_pos,
            door_placed=level.door_placed,
            box_map=level.box_map,
            observation_map=observation_map,
        )
    
    @classmethod
    def load_prefabs(cls, ids):
        return ObservedLevel.stack([ObservedLevel.from_str(prefabs[id]) for id in ids])