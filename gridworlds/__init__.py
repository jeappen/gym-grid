from gym.envs.registration import register


# using max_episode_steps for new API

register(
    id='FourRooms-v0',
    entry_point='gridworlds.envs:FourRooms',
    max_episode_steps=100000,
)    

register(
    id='gridworld-v0',
    entry_point='gridworlds.envs:GridWorld',
    max_episode_steps=100000,
)

register(
    id='PuddleWorld-v0',
    entry_point='gridworlds.envs:PuddleWorld',
    max_episode_steps=100000,
)


register(
    id='PuddleWorldA-v0',
    entry_point='gridworlds.envs:PuddleWorldA',
    max_episode_steps=100000,
)

register(
    id='PuddleWorldB-v0',
    entry_point='gridworlds.envs:PuddleWorldB',
    max_episode_steps=100000,
)

register(
    id='PuddleWorldC-v0',
    entry_point='gridworlds.envs:PuddleWorldC',
    max_episode_steps=100000,
)