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

register(
    id='PuddleWorld-v0',
    entry_point='gridworlds.envs:PuddleWorld',
    timestep_limit=100000,
)

register(
    id='PuddleWorldA-v0',
    entry_point='gridworlds.envs:PuddleWorldA',
    timestep_limit=100000,
)

register(
    id='PuddleWorldB-v0',
    entry_point='gridworlds.envs:PuddleWorldB',
    timestep_limit=100000,
)

register(
    id='PuddleWorldC-v0',
    entry_point='gridworlds.envs:PuddleWorldC',
    timestep_limit=100000,
)