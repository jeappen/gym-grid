"""
Author: Joe

Acknowledgement:
    - The wrappers (BufferedObsEnv, SkipEnv) were originally written by
        Evan Shelhamer and modified by Deepak. Thanks Evan!
    - This file is derived from
        https://github.com/shelhamer/ourl/envs.py
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py
        Deepak Pathak's work
"""
from __future__ import print_function
import numpy as np
from collections import deque
from PIL import Image
from gym.spaces.box import Box
import gym
import time, sys

# Gridworld Tile IDs. Matches with gym PuddleWorld
WORLD_FREE = 0
WORLD_OBSTACLE = 1
WORLD_MINE = 2
WORLD_GOAL = 3
WORLD_FRUIT = 7


class SquareView_grid(gym.ObservationWrapper): 
    """
    Convert observation (row,col) in gridworld to Square View around it
    """
    def __init__(self, env=None, n = None, split_view = None, flatten_mode = None):
        '''
        n = view size around agent
        split_view = Whether tile ID as is or 3 channels, (bad,neutral,good) - one-hot each cell
        flatten_mode = (if split_view)- ways to combine 3 separate channels of gridworld observation. 4 modes for now.
                     0: (-3,1,3), 1: (-3,-1,3), 2: 2 channels (-1*b + 1*g, 1*n) , 3: all channels, as-is
        '''
        super(SquareView_grid, self).__init__(env)
        if(n is None):
            n = 1
        if(split_view is None):
            split_view = False
        if(flatten_mode is None):
            flatten_mode = 0
        self.n = n
        self.split_view = split_view
        self.flatten_mode = flatten_mode
        if split_view and flatten_mode>1: # if flatten_mode > 1 => 2 or 3 channels
            view_size = (1+2*n,1+2*n,flatten_mode)
        else:
            view_size = (1+2*n,1+2*n,1)
        self._view_size = view_size
        view_codes = 10 # number of types of tiles in view
        self.observation_space = Box(low=np.zeros(view_size)-view_codes, high=np.zeros(view_size)+view_codes)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print('step',obs,)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        return obs
        
    def _reset(self):
        obs = self._convert(self.env.reset())
        return obs

    def _render(self, mode='rgb_array', n=None, close=None):
        # TODO: finish this
        if close: return
        view = self.env.unwrapped._get_view(self._convert(self.env.unwrapped.state),self.n,self.split_view)
        # print(view, sep=' ', end='', flush=True)
        return

    def _convert(self, obs):
        view = self.env.unwrapped._get_view(obs,self.n,self.split_view)
        fruit_count = None
        if(self.env.unwrapped.goal_count_dict is not None) and view.ndim ==3: fruit_count = view[0,self.n,self.n]
        # print(view, obs) # 
        if(self.split_view):
            if(self.flatten_mode == 0):
                view = -3*view[0] + 1*view[1] + 3*view[2] #flatten out the view
                view = np.reshape(view,self._view_size)
            elif(self.flatten_mode == 1):
                view = -3*view[0] - 1*view[1] + 3*view[2] #flatten out the view
                view = np.reshape(view,self._view_size)
            elif(self.flatten_mode == 2):
                view = np.array([-1*view[0] + 1*view[2],view[1]]) # 2 channel view
            elif(self.flatten_mode == 3):
                pass #return 3 channels as-is
        if fruit_count is not None: 
            if view.ndim ==3: view[self.n,self.n,:] = fruit_count # replace centre of view with fruit count after modifying view
            else: view[self.n,self.n] = fruit_count # replace centre of view with fruit count after modifying view
        return view

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0], 8: [1.0,1.0,1.0]}

class ColourView_grid(gym.ObservationWrapper):
    """
    Convert observation (row,col) in gridworld to Coloured Square View around it
    """
    def __init__(self, env=None, n = None):
        '''
        n = view size around agent
        '''
        super(ColourView_grid, self).__init__(env)
        if(n is None):
            n = 1
        self.n = n
        view_size = (1+2*n,1+2*n,3)
        self.observation_space = Box(low=0, high=1, shape=view_size)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        return obs
        
    def _reset(self):
        obs = self._convert(self.env.reset())
        return obs
        
    def _render(self, mode='rgb_array', n=None, close=None):
        if close: return
        data = self.env.render(mode=mode, n=self.n)
        return data

    def _convert(self, obs):
        view = self.env.unwrapped._get_colour_view(obs,self.n) # Flat image of colour codes (decided by environment)
        colour_view = np.reshape([COLORS[j] for i in view for j in i],view.shape+(3,))
        return colour_view

class discObs2Box_grid(gym.ObservationWrapper):
    """
    Convert discrete observation in gridworld to Box (X,Y) coordinates.
    """
    def __init__(self, env=None):
        super(discObs2Box_grid, self).__init__(env)
        n = self.env.unwrapped.n
        self.observation_space = Box(low=np.zeros(2), high=np.zeros(2)+n-1)
        # self.env.spec = env.spec

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        return obs

    def _reset(self):
        obs = self._convert(self.env.reset())
        return obs

    def _convert(self, obs):
        return np.array(self.env.unwrapped.ind2coord(obs))


class ChangePerStepReward_grid(gym.RewardWrapper):
    """Change per step reward for gridworld given that free tiles have id = WORLD_FREE"""
    def __init__(self, env=None, per_step=-0.1):
        super(ChangePerStepReward_grid, self).__init__(env)
        self.env.unwrapped.tile_ids[WORLD_FREE] = per_step

    def _reward(self, reward): # To make reward wrapper work
        return reward

class MinesweeperMode(gym.RewardWrapper):
    """Changes Mine and Fruit Rewards. Uses harcoded values hence boo"""
    def __init__(self, env=None, mine_reward = 2 , fruit_reward = -4):
        super(MinesweeperMode, self).__init__(env)
        self.env.unwrapped.tile_ids[WORLD_MINE] = mine_reward
        self.env.unwrapped.tile_ids[WORLD_FRUIT] = fruit_reward

    def _reward(self, reward): # To make reward wrapper work
        return reward


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.

    n is the length of the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations (min=1).

    n.b. first obs is the oldest, last obs is the newest.
         the buffer is zeroed out on reset.
         *must* call reset() for init!
    """
    def __init__(self, env=None, n=4, skip=4, shape=(84, 84),
                    channel_last=True, maxFrames=True):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = maxFrames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = Box(0.0, 255.0, shape)
        self.ch_axis = -1 if channel_last else 0
        self.scale = 1.0 / 255
        self.observation_space.high[...] = 1.0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self._convert(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _convert(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def _rgb2y(self, im):
        """Converts an RGB image to a Y image (as in YUV).

        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""
    def __init__(self, env=None, neg_clip=0.0):
        super(NoNegativeRewardEnv, self).__init__(env)
        self.neg_clip = neg_clip

    def _reward(self, reward):
        new_reward = self.neg_clip if reward < self.neg_clip else reward
        return new_reward


class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def _step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info['steps'] = i + 1
            if done:
                break
        return obs, total_reward, done, info


class MarioEnv(gym.Wrapper):
    def __init__(self, env=None, tilesEnv=False):
        """Reset mario environment without actually restarting fceux everytime.
        This speeds up unrolling by approximately 10 times.
        """
        super(MarioEnv, self).__init__(env)
        self.resetCount = -1
        # reward is distance travelled. So normalize it with total distance
        # https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/lua/super-mario-bros.lua
        # However, we will not use this reward at all. It is only for completion.
        self.maxDistance = 3000.0
        self.tilesEnv = tilesEnv

    def _reset(self):
        if self.resetCount < 0:
            print('\nDoing hard mario fceux reset (40 seconds wait) !')
            sys.stdout.flush()
            self.env.reset()
            time.sleep(40)
        obs, _, _, info = self.env.step(7)  # take right once to start game
        if info.get('ignore', False):  # assuming this happens only in beginning
            self.resetCount = -1
            self.env.close()
            return self._reset()
        self.resetCount = info.get('iteration', -1)
        if self.tilesEnv:
            return obs
        return obs[24:-12,8:-8,:]

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print('info:', info)
        done = info['iteration'] > self.resetCount
        reward = float(reward)/self.maxDistance # note: we do not use this rewards at all.
        if self.tilesEnv:
            return obs, reward, done, info
        return obs[24:-12,8:-8,:], reward, done, info

    def _close(self):
        self.resetCount = -1
        return self.env.close()


class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise"""
    def __init__(self, env=None, percentPad=5):
        super(MakeEnvDynamic, self).__init__(env)
        self.origShape = env.observation_space.shape
        newside = int(round(max(self.origShape[:-1])*100./(100.-percentPad)))
        self.newShape = [newside, newside, 3]
        self.observation_space = Box(0.0, 255.0, self.newShape)
        self.bottomIgnore = 20  # doom 20px bottom is useless
        self.ob = None

    def _observation(self, obs):
        imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
        imNoise[:self.origShape[0]-self.bottomIgnore, :self.origShape[1], :] = obs[:-self.bottomIgnore,:,:]
        self.ob = imNoise
        return imNoise

    # def render(self, mode='human', close=False):
    #     temp = self.env.render(mode, close)
    #     return self.ob
