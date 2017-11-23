# Gridworld environments for Open AI gym

## Maps
* Puddle World ('PuddleWorldB-v0','PuddleWorldST1-v0', 'PuddleWorldST2-v0',etc)
* Mine World ('MineWorldRandomSmall-v0','MineWorldRandomBig-v0')
* Room World ('RoomWorldFinalBig-v0', 'RoomWorldObjectSmall-v0','RoomWorldExit-v0',etc)

and (not changed from starting repo)
* Four Rooms
* Simple Gridworld

### Example Maps
![alt text](https://github.com/yokian/gym-grid/raw/master/img/pw.png "Puddle World")
![alt text](https://github.com/yokian/gym-grid/raw/master/img/mw_small_map.png "Mine World Example")
![alt text](https://github.com/yokian/gym-grid/raw/master/img/ha2t_map.png "Room World Final")

## To install
```
cd gym-grid
pip install -e .
```
## To use
```python
import gym
import gridworlds

env = gym.make('RoomWorld-v0')
```
## To test
```python
cd tests/
test_env.py
```

Recommended to run tests in an python terminal.

## Map Descriptions
- PuddleWorld - 
Avoid centre of map and reach goal (goal position varies depending on map being B,ST1,ST2,etc)
- MineWorld - 
Maximise reward by avoiding Mines and collecting Fruits. Map is randomised at every call of env.reset()
- RoomWorld - 
Complete episode by collecting fruit (to train an 'collect fruit' policy )
- RoomWorldExit - 
Complete episode by reaching gap from random start (to train an 'exit room' policy )
- RoomWorldObjectSmall - 
Complete episode by collecting the fruit and reaching gap from random start (to train an 'collect fruit and exit room' policy )
- RoomWorldFinalBig - 
Complex Map with 6 rooms. Collect all fruits to finish. (Meant as a hierarchical RL benchmark)


## Note
- env_wrapper.py is crucial to get square view output around the agent
- all the env ids are in `gridworlds/__init__.py`
- Roomworlds have the centre of view being number of fruits in a particular room

Starting Point being [this repo](https://github.com/aharutyu/gym-gridworlds)
