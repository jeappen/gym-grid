import gym
import gridworlds
from env_wrapper import discObs2Box_grid,ChangePerStepReward_grid,SquareView_grid

env = gym.make('RoomWorldFinalBig-v0')#('RoomWorldFinal-v0')
we = discObs2Box_grid(env) #wrapped environment
se = SquareView_grid(we,4,True)
# we.env.unwrapped.tile_ids[0] = -.2
