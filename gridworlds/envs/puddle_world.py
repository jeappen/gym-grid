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
WORLD_FRUIT = 7

WORLD_PUDDLE = [4, 5, 6]  # Puddle Codes
puddle_rewards = [-1,-2,-3] # Puddle penalties -1, -2, and -3
puddle_dict = {i:j for i,j in zip(WORLD_PUDDLE,puddle_rewards)}

class PuddleWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, noise=0.0, terminal_reward=10, 
            border_reward=0.0, step_reward=-0.2, start_state_ind=None, wind = 0.5, confusion = 0.1, mine_reward = -4,
            bump_reward =-0.5, fruit_reward = 2, start_states = None,world_file_path = None, init_map = None): #'random'):
        '''
        map = 2D Array with elements indicating type of tile.
        '''
        def load_map(self, fileName):
            theFile = open(fileName, "rb")
            self.map = np.array(pickle.load(theFile))
            self.n = self.map.shape[0]
            theFile.close()
        # Load a map if no init map
        if(init_map is None):
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
        else:
            self.map = init_map
            self.n = self.map.shape[0] # assuming Square shape
        

        self.tile_ids = {WORLD_FREE:step_reward,WORLD_OBSTACLE:bump_reward,WORLD_GOAL:terminal_reward, WORLD_FRUIT: fruit_reward, WORLD_MINE : mine_reward}
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

        # Can speed up finding Goal using np.where()
        goal_locs = np.where(self.map == WORLD_GOAL)
        goal_coords = np.c_[goal_locs]
        self.term_states = [self.coord2ind(c) for c in goal_coords] # allows multiple goal states
        # for i in range(self.n_states-1):
        #     if self.map.T.take(i) == WORLD_GOAL: #T for column wise indexing
        #         self.terminal_state = i # assumes only one goal state.
        #         break
        if (len(self.term_states)>0): self.terminal_state = self.term_states[0] # Picking first one
        else: self.terminal_state = -1
        assert(self.terminal_state is not None)

        # self.terminal_state = self.n_states - 2 - terminal_state_offset
        self.absorbing_state = self.n_states - 1
        self.done = False

        if start_states is None:
            self.start_states = [[6, 1], [7, 1], [11, 1], [12, 1]]
            self.start_state_ind = start_state_ind
            # TODO: allow, random start states?
        

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

        if np.random.rand() < self.noise: # Randomly pick an action
            action = self.action_space.sample()
        
        if(np.random.rand() < self.confusion):  # if confused, then pick action apart from that specified
            rand_act = self.action_space.sample()
            while rand_act == action: # while action is the same as picked, keep sampling
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

        # Set self.done at end of step
        if self.state == self.terminal_state or self.state in self.term_states:
            self.done = True
            return self.state, self._get_reward(), self.done, None

        return self.state, reward, self.done, None

    def _get_view(self, state=None, n=None, split_view = None):
        # get view of n steps around
        # input: state: (row,col)

        if(state is None):
            state = self.state
        if(n is None):
            n = 1
        if(split_view is None):
            split_view = False

        row,col = state

        up = max(row - n, 0)
        down = min(row + n, self.n - 1)
        left = max(col - n, 0)
        right = min(col + n, self.n - 1)

        view_patch = self.map[up:down+1, left:right+1]

        view = np.zeros((2*n+1,2*n+1))
        view_up = max(0, n-row)
        view_down = min(self.n -1 - row + n,2*n)
        view_left = max(0, n-col)
        view_right = min(self.n -1 - col + n,2*n)

        view[view_up:view_down+1, view_left:view_right+1] = view_patch # handles cases where n size gives window off the map

        # modify view here (different channels, color-code, etc)
        # Can divide into three channels. 1* to make it 0-1
        bad_c = 1*np.any([(view == x) for x in WORLD_PUDDLE+[WORLD_MINE]],axis=0)
        good_c = 1*(view == [WORLD_GOAL,WORLD_FRUIT])
        neutral_c = 1*(view == WORLD_OBSTACLE)
        new_view = -1*bad_c + 1*neutral_c + 2*good_c # can return this
        view_channels = np.array([bad_c,neutral_c,good_c]) # or this without loss of generality

        if(split_view):
            return view_channels
        else:
            return view

    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward

        tile = self.map.T.take(new_state)
        reward = self.tile_ids[tile] # Use the reward dictionary to give reward based on tile

        r,c = self.ind2coord(new_state)
        if(tile == WORLD_FRUIT or tile == WORLD_MINE): self.map[r,c] = WORLD_FREE # "pickup fruits" and "step on Mines" 

        # reward = self.step_reward

        # if self.border_reward != 0 and self.at_border():
        #     reward = self.border_reward

        #Uncomment to add bump-reward
        # if self.bump_reward != 0 and self.state == new_state: 
        #     reward = self.bump_reward

        return reward

    def change_reward(self, step_reward = None, bump_reward = None, terminal_reward = None):
        # For easy change of step_reward,etc
        if(step_reward is None):
            step_reward = self.step_reward
        if(bump_reward is None):
            bump_reward = self.bump_reward
        if(terminal_reward is None):
            terminal_reward = self.terminal_reward
        self.tile_ids = {WORLD_FREE:step_reward,WORLD_OBSTACLE:bump_reward,WORLD_GOAL:terminal_reward}
        self.tile_ids.update(puddle_dict)
        pass

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

class PuddleWorld_st1(PuddleWorld):
# puddle world sub task 1
    def __init__(self):
        super(PuddleWorld_st1, self).__init__(world_file_path="PW_st1.dat")

class PuddleWorld_st2(PuddleWorld):
# puddle world sub task 2
    def __init__(self):
        super(PuddleWorld_st2, self).__init__(world_file_path="PW_st2.dat")

class PuddleWorld_a2t(PuddleWorld):
# puddle world as in a2t paper
    def __init__(self):
        super(PuddleWorld_a2t, self).__init__(world_file_path="PW_a2t.dat")

class PuddleWorld_random(PuddleWorld):
# puddle world w/random fruits
    def __init__(self, n = None,scaling = None):
        if(n is None):
            self.n = 14
        else: self.n = n
        if scaling is None:
            self.scaling = {'fruits':3,'mines':3}
        else: self.scaling = scaling
        m = self.load_random_map()
        super(PuddleWorld_random, self).__init__( init_map = m)
    
    def load_random_map(self):
        m = np.zeros((self.n,self.n))
        num_fruits = self.scaling['fruits']*self.n;
        num_mines = self.scaling['mines']*self.n;
        random_states = np.random.choice(self.n**2,num_fruits+num_mines,replace=False)
        rw = random_states%self.n
        cl = random_states//self.n
        f_ind = list(zip(rw,cl))[:num_fruits]
        m_ind = list(zip(rw,cl))[num_fruits:]

        for i,j in f_ind: m[i,j] = WORLD_FRUIT
        for i,j in m_ind: m[i,j] = WORLD_MINE
        m[0,:] =  m[-1,:] =  m[:,0] = m[:,-1] = WORLD_OBSTACLE # Make Walls
        return m
    def reload_random(self):
        m = self.load_random_map()
        self.map = m