import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pickle,os

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

WORLD_FREE = 0
WORLD_OBSTACLE = 1
WORLD_MINE = 2
WORLD_GOAL = 3

WORLD_PUDDLE = [4, 5, 6]  # Puddle Codes
puddle_rewards = [-1,-2,-3] # Puddle penalties -1, -2, and -3
puddle_dict = {i:j for i,j in zip(WORLD_PUDDLE,puddle_rewards)}

class PuddleWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=14, noise=0.0, terminal_reward=10, 
            border_reward=0.0, step_reward=-0.1, start_state_ind=None, wind = 0.5, confusion = 0,
            bump_reward =-0.5, start_states = None,world_file_path = None): #'random'):
        '''
        map = 2D Array with elements indicating type of tile.
        '''
        def load_map(self, fileName):
            theFile = open(fileName, "rb")
            self.map = np.array(pickle.load(theFile))
            self.n = self.map.shape[0]
            theFile.close()
        # Load a map 
        assert(world_file_path is not None)
        if world_file_path is not None:
            if not os.path.exists(world_file_path):
                # Now search the saved_maps folder
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "saved_maps", world_file_path)
                if os.path.exists(rel_path):
                    world_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % world_file_path)
            load_map(self,world_file_path)
            print("\nFound Saved Map\n")
        

        self.tile_ids = {WORLD_FREE:step_reward,WORLD_OBSTACLE:bump_reward,WORLD_GOAL:terminal_reward}
        self.tile_ids.update(puddle_dict)

        # self.n = n # Uncomment when not loading map
        self.noise = noise
        self.confusion = confusion
        self.terminal_reward = terminal_reward
        self.border_reward = border_reward
        self.bump_reward = bump_reward
        self.step_reward = step_reward
        self.n_states = self.n ** 2 + 1
        self.terminal_state = None
        for i in range(self.n_states-1):
            if self.map.T.take(i) == WORLD_GOAL: #T for column wise indexing
                self.terminal_state = i # assumes only one goal state.
                break
        assert(self.terminal_state is not None)

        # self.terminal_state = self.n_states - 2 - terminal_state_offset
        self.absorbing_state = self.n_states - 1
        self.done = False

        if start_states is None:
            self.start_states = [[6, 1], [7, 1], [11, 1], [12, 1]]
            self.start_state_ind = start_state_ind
        

        # Simulation related variables
        self._reset()
        self._seed()

        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=np.zeros(2), high=np.zeros(2)+n-1) # use wrapper instead
        self.observation_space = spaces.Discrete(self.n_states) # with absorbing state
        #self._seed()

    def _step(self, action):
        assert self.action_space.contains(action)

        if self.state == self.terminal_state:
            self.state = self.absorbing_state #Careful now, don't run env. without resetting
            self.done = True
            return self.state, self._get_reward(), self.done, None

        [row, col] = self.ind2coord(self.state)

        if np.random.rand() < self.noise: # Randomnly pick an action
            action = self.action_space.sample()
        
        if(np.random.rand() < self.confusion):  # if confused, then pick action apart from that specified
            rand_act = self.action_space.sample()
            while rand_act == action:
                rand_act = self.action_space.sample()
            action = rand_act

        if action == UP:
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n - 1)
        elif action == LEFT:
            col = max(col - 1, 0)

        new_state = self.coord2ind([row, col])

        # Check if new state is an obstacle
        if(self.map.T.take(new_state) == WORLD_OBSTACLE):
            new_state = self.state # State remains unchanged

        reward = self._get_reward(new_state=new_state)

        self.state = new_state

        return self.state, reward, self.done, None

    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward

        reward = self.tile_ids[self.map.T.take(new_state)] # Use the reward dictionary to give reward based on tile

        # reward = self.step_reward

        # if self.border_reward != 0 and self.at_border():
        #     reward = self.border_reward

        # if self.bump_reward != 0 and self.state == new_state:
        #     reward = self.bump_reward

        return reward

    def at_border(self):
        [row, col] = self.ind2coord(self.state)
        return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

    def ind2coord(self, index):
        assert(index >= 0)
        #assert(index < self.n_states - 1)

        col = index // self.n
        row = index % self.n

        return [row, col]


    def coord2ind(self, coord):
        [row, col] = coord
        assert(row < self.n)
        assert(col < self.n)

        return col * self.n + row


    def _reset(self):
        if(self.start_state_ind is None):
            start_state_ind = np.random.randint(len(self.start_states))
        else:
            start_state_ind = self.start_state_ind
        self.start_state = self.coord2ind(self.start_states[start_state_ind])
        self.state = self.start_state #if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 1)
        self.done = False
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        pass
      
class PuddleWorldA(PuddleWorld):

    def __init__(self):
        super(PuddleWorldA, self).__init__(world_file_path="PuddleWorldA.dat")

class PuddleWorldB(PuddleWorld):

    def __init__(self):
        super(PuddleWorldB, self).__init__(world_file_path="PuddleWorldB.dat")

class PuddleWorldC(PuddleWorld):

    def __init__(self):
        super(PuddleWorldC, self).__init__(world_file_path="PuddleWorldC.dat")