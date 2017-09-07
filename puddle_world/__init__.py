from gym.envs.registration import register

register(
    id='FourRooms-v0',
    entry_point='gridworlds.envs:FourRooms',
    timestep_limit=100000,
)    

register(
    id='gridworld-v0',
    entry_point='gridworlds.envs:GridWorld',
    timestep_limit=100000,
)
