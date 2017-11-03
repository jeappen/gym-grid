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

register(
    id='PuddleWorldST1-v0',
    entry_point='gridworlds.envs:PuddleWorld_st1',
    max_episode_steps=100000,
)

register(
    id='PuddleWorldST2-v0',
    entry_point='gridworlds.envs:PuddleWorld_st2',
    max_episode_steps=100000,
)

register(
    id='PuddleWorldA2T-v0',
    entry_point='gridworlds.envs:PuddleWorld_a2t',
    max_episode_steps=100000,
)

register(
    id='MineWorldRandomSmall-v0',
    entry_point='gridworlds.envs:PuddleWorld_random',
    max_episode_steps=30,
    kwargs = {'n':14}
)

register(
    id='MineWorldRandomBig-v0',
    entry_point='gridworlds.envs:PuddleWorld_random',
    max_episode_steps=60,
    kwargs = {'n':28}
)

register(
    id='RoomWorld-v0',
    entry_point='gridworlds.envs:RoomWorld',
    max_episode_steps=100000,
    kwargs = {'n':14}
)

register(
    id='RoomWorldExit-v0',
    entry_point='gridworlds.envs:RoomWorld',
    max_episode_steps=100000,
    kwargs = {'n':14,'mode':'exit'}
)

register(
    id='RoomWorldObjectSmall-v0',
    entry_point='gridworlds.envs:RoomWorldObject',
    max_episode_steps=100000,
    kwargs = {'n':14}
)

register(
    id='RoomWorldObjectBig-v0',
    entry_point='gridworlds.envs:RoomWorldObject',
    max_episode_steps=100000,
    kwargs = {'n':28}
)

register(
    id='RoomWorldFinalBig-v0',
    entry_point='gridworlds.envs:RoomWorldFinal',
    max_episode_steps=100000
)