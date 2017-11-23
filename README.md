# Gridworld environments for Open AI gym

* Puddle World
![alt text](https://github.com/yokian/gym-grid/raw/master/img/pw.png "Puddle World")
* Mine World
![alt text](https://github.com/yokian/gym-grid/raw/master/img/mw_small_map.png "Mine World Example")
* Room World
![alt text](https://github.com/yokian/gym-grid/raw/master/img/ha2t_map.png "Room World Final")

and
* Four Rooms
* Simple Gridworld

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
```

Run test_env.py.
Note : env_wrapper.py is crucial to get square view output


Starting Point being [this repo](https://github.com/aharutyu/gym-gridworlds)
