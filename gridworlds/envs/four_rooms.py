import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

""" Four rooms. The goal is either in the 3rd room, or in a hallway adjacent to it
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_ROOMS = 4

class FourRooms(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.room_sizes = [[5,5], [6,5], [5,5], [4,5]]
    self.pre_hallways = [ 
                          { tuple([2,4]) : [RIGHT, 0], tuple([4,1]) : [DOWN, 3]},
                          { tuple([2,0]) : [LEFT, 0], tuple([5,2]) : [DOWN, 1]},
                          { tuple([0,2]) : [UP, 1], tuple([2,0]) : [LEFT, 2]},
                          { tuple([3,4]) : [RIGHT, 2], tuple([0,1]) : [UP, 3]},
                        ]
    self.hallway_coords = [ [2,5], [6,2], [2,-1], [-1,1] ]
    self.hallways = [ #self.hallways[i][j] = [next_room, next_coord] when taking action j from hallway i#
                      [ [0, self.hallway_coords[0]], [1, [2,0]], [0, self.hallway_coords[0]], [0, [2,4]] ],
                      [ [1, [5,2]], [1, self.hallway_coords[1]], [2, [0,2]], [1, self.hallway_coords[1]] ],
                      [ [2, self.hallway_coords[2]], [2, [2,0]], [2, self.hallway_coords[2]], [3, [3,4]] ],
                      [ [0, [4,1]], [3, self.hallway_coords[3]], [3, [0,1]], [3, self.hallway_coords[3]] ]
                    ]

    
    self.offsets = [0] * (NUM_ROOMS + 1)
    for i in range(NUM_ROOMS):
      self.offsets[i + 1] = self.offsets[i] + self.room_sizes[i][0] * self.room_sizes[i][1] + 1
    self.n_states = self.offsets[4] + 1
    self.absorbing_state = self.n_states

    self.goal = [2, [1, 2]]
    self.terminal_state = self.encode(self.goal)

    self.noise = 0 #0.33
    self.step_reward = 0.0
    self.terminal_reward = 1.0
    self.bump_reward = -0.1

    # start state random location in start room
    start_room = 0
    sz = self.room_sizes[start_room]
    self.start_state = self.offsets[start_room] + np.random.randint(sz[0]*sz[1] - 1)
    self._reset()

    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Discrete(self.n_states) # with absorbing state


  def ind2coord(self, index, sizes=None):
    if sizes is None:
      sizes = [self.n]*2
    [rows, cols] = sizes

    assert(index >= 0)

    row = index // cols
    col = index % cols

    return [row, col]


  def coord2ind(self, coord, sizes=None):
    if sizes is None:
      sizes = [self.n]*2

    [rows, cols] = sizes
    [row, col] = coord

    assert(row < rows)
    assert(col < cols)

    return row * cols + col

  def in_hallway_index(self, index=None):
    if index is None:
      index = self.state
    return index in [offset - 1 for offset in self.offsets]

  def in_hallway_coord(self, coord):
    return coord in self.hallway_coords

  def encode(self, location, in_hallway=None):
    [room, coord] = location
    if in_hallway is None:
      in_hallway = self.in_hallway_coord(coord)

    if in_hallway:
      return self.offsets[room + 1] - 1
      # maybe have hallways as input
    ind_in_room = self.coord2ind(coord, sizes=self.room_sizes[room])
    return ind_in_room + self.offsets[room]

  def decode(self, index, in_hallway=None):
    if in_hallway is None:
      in_hallway = self.in_hallway_index(index=index)

    room = [r for r, offset in enumerate(self.offsets[1:5]) if index < offset][0]
    if in_hallway:
      coord_in_room = self.hallway_coords[room]
    else:
      coord_in_room = self.ind2coord(index - self.offsets[room], sizes=self.room_sizes[room])
    return room, coord_in_room # hallway

  def _step(self, action):
    assert self.action_space.contains(action)

    if self.state == self.terminal_state:
      self.state = self.absorbing_state
      self.done = True
      return self.state, self._get_reward(), self.done, None

    in_hallway = self.in_hallway_index()
    [room, coord]= self.decode(self.state, in_hallway=in_hallway)
    room2 = room; coord2 = coord

    if np.random.rand() < self.noise:
      action = self.action_space.sample()

    if in_hallway: # hallway action
      [room2, coord2] = self.hallways[room][action]
    
    elif tuple(coord) in self.pre_hallways[room].keys():
      hallway_info = self.pre_hallways[room][tuple(coord)]
      if action == hallway_info[0]:
        room2 = hallway_info[1]
        coord2 = self.hallway_coords[room2]
    
    else: # normal action
      [row, col] = coord
      [rows, cols] = self.room_sizes[room]
      if action == UP:
        row = max(row - 1, 0)
      elif action == DOWN:
        row = min(row + 1, rows - 1)
      elif action == RIGHT:
        col = min(col + 1, cols - 1)
      elif action == LEFT:
        col = max(col - 1, 0)
      coord2 = [row, col]

    new_state = self.encode([room2, coord2])
    self.state = new_state

    reward = self._get_reward(new_state=new_state)

    return new_state, reward, self.done, None




  def _get_reward(self, new_state=None):
    if self.done:
      return self.terminal_reward

    reward = self.step_reward
    
    if self.bump_reward != 0 and self.state == new_state:
      reward = self.bump_reward

    return reward

  def at_border(self):
    [row, col] = self.ind2coord(self.state)
    return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)


  def _reset(self):
    self.state = self.start_state if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 1)
    self.done = False
    return self.state

  def _render(self, mode='human', close=False):
    pass
      
