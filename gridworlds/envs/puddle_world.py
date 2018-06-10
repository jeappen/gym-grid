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
WORLD_FRUIT = 7 # Small positive reward, non-terminal state
WORLD_INVISIBLE_GOAL = 8 # Terminal state not visible to agent when using get_view()

WORLD_PUDDLE = [4, 5, 6]  # Puddle Codes
puddle_rewards = [-1,-2,-3] # Puddle penalties -1, -2, and -3
puddle_dict = {i:j for i,j in zip(WORLD_PUDDLE,puddle_rewards)} # To map rewards associated with Puddles

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0],8: [1.0,1.0,1.0]}

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
        # Load a map if no init map provided
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

        # To map tiles and rewards associated
        self.tile_ids = {WORLD_FREE:step_reward,WORLD_OBSTACLE:bump_reward,WORLD_GOAL:terminal_reward,\
         WORLD_FRUIT: fruit_reward, WORLD_MINE : mine_reward, WORLD_INVISIBLE_GOAL: terminal_reward}
        self.tile_ids.update(puddle_dict)

        # To map colours to tile IDs
        self.tile_colour_ids = {WORLD_FREE:0,WORLD_OBSTACLE:1,WORLD_GOAL:3,\
         WORLD_FRUIT: 2, WORLD_MINE : 4, WORLD_INVISIBLE_GOAL: 0}

        # Handling fruit count when required. Needed for Nose in multiple room maps
        try:
            self.num_rooms # Does num_rooms exist?
        except (AttributeError,NameError):
            self.num_rooms = None
        try:
            self.room_map # Does room_map exist?
        except (AttributeError,NameError):
            self.room_map = None
        try:
            self.goal_count_dict # Does goal_count_dict exist?
        except (AttributeError,NameError):
            self.goal_count_dict = None
        

        # self.n = n # Uncomment when not loading map
        self.noise = noise
        self.confusion = confusion
        self.terminal_reward = terminal_reward
        self.border_reward = border_reward
        self.bump_reward = bump_reward
        self.step_reward = step_reward
        self.n_states = self.n ** 2 + 1
        self.terminal_state = None

        self.set_term_state() # searches map and sets terminal states

        # self.terminal_state = self.n_states - 2 - terminal_state_offset
        self.absorbing_state = self.n_states - 1
        self.done = False

        self.set_start_state(start_states, start_state_ind)        

        # Simulation related variables
        self._reset()
        self._seed()

        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=np.zeros(2), high=np.zeros(2)+n-1) # use wrapper instead
        self.observation_space = spaces.Discrete(self.n_states) # with absorbing state
        #self._seed()

    def set_term_state(self):
        # searches map and sets terminal states
        goal_locs = np.where((self.map == WORLD_GOAL) + (self.map == WORLD_INVISIBLE_GOAL))
        goal_coords = np.c_[goal_locs]
        self.term_states = [self.coord2ind(c) for c in goal_coords] # allows multiple goal states
        
        if (len(self.term_states)>0): self.terminal_state = self.term_states[0] # Picking first one
        else: self.terminal_state = -1
        assert(self.terminal_state is not None)

    def set_start_state(self, start_states = None, start_state_ind = None):
        self.start_state_ind = start_state_ind
        if start_states is None:
            self.start_states = [[6, 1], [7, 1], [11, 1], [12, 1]]
        elif start_states ==[]: # random start states hack
            candidate_starts = np.where(self.map != WORLD_OBSTACLE)
            start_coords = np.c_[candidate_starts]
            self.start_states = [c for c in start_coords] # picks ALL states apart from obstacles
        else:
            self.start_states = start_states

    def _step(self, action):
        assert self.action_space.contains(action)
        info = {}

        if self.state == self.terminal_state:
            self.state = self.absorbing_state # Careful now, don't run env. without resetting
            self.done = True
            return self.state, self._get_reward(), self.done, info

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
            return self.state, self._get_reward(), self.done, info

        return self.state, reward, self.done, info

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

        # if self.done: # Skip if done
        #     return view

        if(split_view):
            # modify view here (different channels, color-code, etc)
            # Can divide into three channels. 1* to make it 0-1
            bad_l = WORLD_PUDDLE+[WORLD_MINE]
            good_l = [WORLD_GOAL,WORLD_FRUIT]
            bad_c = 1*np.any([(view == x) for x in bad_l],axis=0)
            good_c = 1*np.any([(view == x) for x in good_l],axis=0)
            neutral_c = 1*(view == WORLD_OBSTACLE)
            new_view = -1*bad_c + 1*neutral_c + 2*good_c # can return this
            view_channels = np.array([bad_c,neutral_c,good_c]) # or this without loss of generality
            return_view = view_channels
        else:
            return_view = view

        if(self.num_rooms is not None and not self.done): # If num_rooms is >1 then replace centre with fruit count in each view
            num_fruits = self.goal_count_dict[self.room_map[row,col]]
            if(split_view):
                return_view[:,n,n] = num_fruits
            else:
                return_view[n,n] = num_fruits

        return return_view

    def _get_colour_view(self, state=None, n=None):
        ''' Gets colour codes for environment objects.
            Lets you define colour in the environment wrapper.
        '''
        view = self._get_view(state,n,False)    # get raw view without splitting into good, bad channels
        if(self.num_rooms is not None):     # This means centre of view is num of fruits
            view[n,n] = min(view[n,n],3)    # Let 3 be cutoff (Since there's no colour for 4!)
        colour_view = np.reshape([self.tile_colour_ids[j] for i in view for j in i],view.shape)
        return colour_view    
        
    def render(self, mode='rgb_array', n=None, close=None):
        if close: return
        if n is None:
            n = 2
        if mode == 'rgb_array':
            data = self._get_colour_view(self.state, n )
            # Coded image
            return data
        return None

    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward

        tile = self.map.T.take(new_state)
        reward = self.tile_ids[tile] # Use the reward dictionary to give reward based on tile

        r,c = self.ind2coord(new_state)

        self.found_fruit_in_last_turn = (tile == WORLD_FRUIT) # To reduce counter for the Roomworld

        if(tile == WORLD_FRUIT or tile == WORLD_MINE): self.map[r,c] = WORLD_FREE # "pickup fruits" and "step on Mines" 

        # reward = self.step_reward # Commented out to make it easier to infer tile from reward ( change tile_id[WORLD_FREE] before uncommenting this)

        # if self.border_reward != 0 and self.at_border():
        #     reward = self.border_reward

        # Uncomment to add bump-reward
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
        new_tile_ids = {WORLD_FREE:step_reward,WORLD_OBSTACLE:bump_reward,WORLD_GOAL:terminal_reward, WORLD_INVISIBLE_GOAL:terminal_reward}
        self.tile_ids.update(new_tile_ids)
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
        if(self.start_state_ind is None): # i.e. if start state is not fixed
            start_state_ind = np.random.randint(len(self.start_states))
        else:
            start_state_ind = self.start_state_ind
        # print(self.start_states,start_state_ind)
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
# puddle world w/random fruits. No terminal state (should stop after a few steps)
    def __init__(self, n = None, objects = None):
        # objects: to fix number and ratio of fruits to mines in the room (sum of terms <= n**2 . Limited by size of room)
        if(n is None):
            self.n = 14
        else: self.n = n
        if objects is None:
            self.objects = {'fruits':3*self.n,'mines':3*self.n}
        else: self.objects = objects
        m = self.load_random_map()
        super(PuddleWorld_random, self).__init__( init_map = m)
    
    def load_random_map(self):
        m = np.zeros((self.n,self.n))
        num_fruits = self.objects['fruits'];
        num_mines = self.objects['mines'];
        random_states = np.random.choice(self.n**2,num_fruits+num_mines,replace=False)
        rw = random_states%self.n
        cl = random_states//self.n
        f_ind = list(zip(rw,cl))[:num_fruits]
        m_ind = list(zip(rw,cl))[num_fruits:]

        for i,j in f_ind: m[i,j] = WORLD_FRUIT
        for i,j in m_ind: m[i,j] = WORLD_MINE
        m[0,:] =  m[-1,:] =  m[:,0] = m[:,-1] = WORLD_OBSTACLE # Make Walls (can overwrite fruits and mines)
        return m

    def reload_random(self):
        m = self.load_random_map()
        self.map = m

    
    def _reset(self):
        #Randomising map at each run
        self.reload_random(); 
        return super(PuddleWorld_random, self)._reset()

class RoomWorld(PuddleWorld):
# Bounded 2 Rooms w/exit to train sub-policies
    def __init__(self, n = None, objects = None, mode = None):
        # mode : 'fruit' - learn to pick up fruit, 'exit' - learn to exit room
        if(n is None): # n >= 5
            self.n = 14
        else: self.n = n
        
        if objects is None:
            self.objects = {'fruits':1,'mines':0} # make sure this is small enough to fit inside world
        else: self.objects = objects

        if(mode is None): # n >= 5
            self.mode = 'fruit'
        else: self.mode = mode
        self.num_rooms = 2 # to make room indices
        
        m,(i,j) = self.load_random_map()

        if self.mode == 'fruit':
            start_states = [[i,j]] # Start from the gap and find the fruit
        else: start_states = [] # Start from a random free location and find the gap (invisible goal)
        # print(start_states)
        super(RoomWorld, self).__init__(init_map = m, start_states = start_states)
    
    def assign_fruit_locations(self, m):
        free_locs = np.where(m == WORLD_FREE)
        free_coords = np.c_[free_locs]
        free_states = free_coords#[self.coord2ind(c) for c in free_coords] # picks all free states
        num_fruits = self.objects['fruits']
        self.num_fruits_left = num_fruits # initialize this
        num_mines = self.objects['mines']
        # throws error if too many mines+fruits
        random_states = np.random.choice(len(free_states),num_fruits+num_mines,replace=False) 
        candidate_states = [free_states[s] for s in random_states]       
        f_ind = candidate_states[:num_fruits]
        m_ind = candidate_states[num_fruits:]
        for k,l in f_ind: m[k,l] = WORLD_FRUIT 
        for k,l in m_ind: m[k,l] = WORLD_MINE
        return m

    def make_room_map(self, m, j):
        # Nose for fruit, can know number of fruits in the room

        ## first make index of room map
        room_map = np.ones(m.shape)*-1
        room_map[m!=WORLD_OBSTACLE] = 0
        dummy_map = np.hstack([np.ones((m.shape[0],j)),2*np.ones((m.shape[0],self.n-j))])
        room_map[room_map==0] = dummy_map[room_map==0] 

        return room_map

    def make_goal_count_dict(self, m, room_map):
        ## Now make a count for each room index
        goal_count = room_map[m==WORLD_FRUIT] # eg [1,2,2,3] means 1 goal in room 1, 2 in room 2 and 1 in room 3
        self.goal_count_dict = {i+1:0 for i in range(self.num_rooms)}
        self.goal_count_dict[-1] = 0 # To catch exception where obstacle is centered in view
        for r in goal_count:
            self.goal_count_dict[r] += 1


    def load_random_map(self):
        # Returns random map with two rooms and the gap between them
        m = np.zeros((self.n,self.n))
        m[0,:] =  m[-1,:] =  m[:,0] = m[:,-1] = WORLD_OBSTACLE # Make Walls
        i,j =  np.random.randint(1,self.n-1),np.random.randint(2,self.n-2) # pick random row and col to make exit between rooms
        m[:,j] = WORLD_OBSTACLE # Makes intersecting wall
        m[i,j] = WORLD_FREE # Makes gap between rooms
        if self.mode == 'fruit':
            m = self.assign_fruit_locations(m)
        else: # Assumes learn2exit mode otherwise
            # Makes invisible goal in gap between rooms.
            # Invisible so that agent learns to see the gap structure
            m[i,j] = WORLD_INVISIBLE_GOAL
        room_map = self.make_room_map(m,j)
        self.make_goal_count_dict(m, room_map)

        if np.random.randint(2): # like flipping a coin (bernoulli(0.5))
            m = m.T # transpose the Map to learn vertical representations as well.
            temp = i # swap i,j to keep gap location correct
            i = j
            j = temp
            room_map = room_map.T

        self.gap_i = i
        self.gap_j = j
        self.room_map = room_map

        return m,[i,j]

    def reload_random(self):
        m,[i,j] = self.load_random_map()
        self.map = m
        if self.mode == 'fruit':
            start_states = [[i,j]]
            # Now to enable termination after finding the fruit change to WORLD_GOAL. 
            # This hinges on _get_view() aliasing fruits and goals as 'good'
            self.map[self.map==WORLD_FRUIT] = WORLD_GOAL 
        else:
            start_states = []
        self.set_start_state(start_states,self.start_state_ind)
        self.set_term_state()

    def _step(self, action):
        return_val = super(RoomWorld, self)._step(action) # state, reward, done, _
        if not return_val[2]:
            [row, col] = self.ind2coord(return_val[0]) 
            self.goal_count_dict[self.room_map[row,col]] -= self.found_fruit_in_last_turn # Reduce room index fruit counter if fruit was found
        return return_val
    
    def _reset(self):
        #Randomising map at each run
        self.reload_random(); 
        return super(RoomWorld, self)._reset()

class RoomWorldObject(RoomWorld):
    ''' Bounded 2 Rooms w/exit. Need to pick up all fruits and reach gap to complete task
    Now solvable since room fruit count is part of observation 
    Hard task for large n! Without a non-markovian policy, will need to square view large 
    (to keep fruits in view, thus the agent realising there's work to be done before leaving) '''

    def _step(self, action): # To set goal once all fruits are taken
        # First take care of room index fruit counter
        return_val = super(RoomWorldObject, self)._step(action) # state, reward, done, _
        self.num_fruits_left -= self.found_fruit_in_last_turn # Reduce total fruit counter if fruit was found
        if self.num_fruits_left <= 0: # set goal state to gap if no fruits in map
            self.map[self.gap_i, self.gap_j] = WORLD_INVISIBLE_GOAL
            self.set_term_state() # Refresh terminal state list after adding goal
        return return_val

    def reload_random(self): # redefine to stop fruit from being goal
        m,[i,j] = self.load_random_map()
        self.map = m
        if self.mode == 'fruit':
            start_states = [[i,j]]
        else:
            start_states = []
        self.set_start_state(start_states,self.start_state_ind)
        self.set_term_state()

class RoomWorldObjectFixed(RoomWorld):
    ''' Bounded 2 Rooms w/exit. Same as before but not random at each run '''

    def _step(self, action): # To set goal once all fruits are taken
        # First take care of room index fruit counter
        return_val = super(RoomWorldObjectFixed, self)._step(action) # state, reward, done, _
        self.num_fruits_left -= self.found_fruit_in_last_turn # Reduce total fruit counter if fruit was found
        if self.num_fruits_left <= 0: # set goal state to gap if no fruits in map
            self.map[self.gap_i, self.gap_j] = WORLD_INVISIBLE_GOAL
            self.set_term_state() # Refresh terminal state list after adding goal
        return return_val

    def reload_random(self): # redefine to stop fruit from being goal
        m,[i,j] = self.load_random_map()
        self.map = m
        if self.mode == 'fruit':
            start_states = [[i,j]]
        else:
            start_states = []
        self.set_start_state(start_states,self.start_state_ind)
        self.set_term_state()

    def _reset(self):
        # Clear up existing fruits
        self.map[self.map==WORLD_FRUIT] = WORLD_FREE
        # No longer Randomising map at each run
        self.map = self.assign_fruit_locations(self.map)
        # Reset Invisible GOal
        self.map[self.gap_i, self.gap_j] = WORLD_FREE
        self.make_goal_count_dict(self.map, self.room_map)
        self.set_start_state([[self.gap_i, self.gap_j]],self.start_state_ind)
        self.set_term_state()
        return super(RoomWorld, self)._reset()

class RoomWorldFinal(PuddleWorld):
    ''' Set of 6 rooms. Need to pick up all fruits and reach gap to complete task
    Hardest task for large n! Useful as simple HRL testbed.'''
    def __init__(self, n = None):
        if(n is None): # n >= 5
            self.n = 32
        else: self.n = n
        
        m = self.load_map()

        start_states = [[30,14],[30,15],[30,16]]

        super(RoomWorldFinal, self).__init__(init_map = m, start_states = start_states)
    
    def load_map(self):
        # Returns harcoded map. TODO: Make this smaller?
        m = np.zeros((self.n,self.n))

        fruit_indexes = [[2,2],[24,2],[1,15],[2,27],[23,27]]
        gap_indexes = [[16,10],[16,21],[8,10],[8,21],[5,15]]

        # Make walls
        m[:,10] = m[:,21] = WORLD_OBSTACLE
        m[10,:11] = m[10,21:] = WORLD_OBSTACLE
        m[5,10:21] = WORLD_OBSTACLE
        m[0,:] =  m[-1,:] =  m[:,0] = m[:,-1] = WORLD_OBSTACLE # Make Surrounding walls

        # Make gaps
        for i,j in gap_indexes:
            m[i,j] = WORLD_FREE

        # Set fruits
        for i,j in fruit_indexes:
            m[i,j] = WORLD_FRUIT
        
        self.num_fruits_left = len(fruit_indexes)
        self.num_rooms = 6

        # Nose for fruit, can know number of fruits in the room

        ## first make index of room map
        room_map = np.ones(m.shape)*-1
        room_map[m!=WORLD_OBSTACLE] = 0
        room_map[:10,:11] = 1
        room_map[10:,:11] = 2
        room_map[:6,11:21] = 3
        room_map[6:,11:21] = 4
        room_map[:10,21:] = 5
        room_map[10:,21:] = 6
        room_map[m==WORLD_OBSTACLE] = -1

        self.room_map = room_map

        ## Now make a count for each room index
        goal_count = room_map[m==WORLD_FRUIT] # eg [1,2,2,3] means 1 goal in room 1, 2 in room 2 and 1 in room 3
        self.goal_count_dict = {i+1:0 for i in range(self.num_rooms)}
        self.goal_count_dict[-1] = 0 # To catch exception where obstacle is centered in view
        for r in goal_count:
            self.goal_count_dict[r] += 1 

        return m

    def _step(self, action):
        return_val = super(RoomWorldFinal, self)._step(action) # state, reward, done, _
        if not return_val[2]:
            [row, col] = self.ind2coord(return_val[0])
            self.goal_count_dict[self.room_map[row,col]] -= self.found_fruit_in_last_turn # Reduce room index fruit counter if fruit was found
        self.num_fruits_left -= self.found_fruit_in_last_turn # Reduce fruit counter if fruit was found
        if self.num_fruits_left <= 1: # set goal state if one fruit left in map
            self.map[self.map == WORLD_FRUIT] = WORLD_GOAL
            self.set_term_state() # Refresh terminal state list after adding goal
        return return_val