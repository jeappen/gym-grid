import gym
import gridworlds
from env_wrapper import discObs2Box_grid,ChangePerStepReward_grid

env = gym.make('PuddleWorldB-v0')
we = discObs2Box_grid(env) #wrapped environment
we.env.unwrapped.tile_ids[0] = -.2
