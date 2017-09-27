import gym
import gridworlds
from env_wrapper import discObs2Box_grid
from collections import defaultdict
import numpy as np

class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        # if not isinstance(observation_space, discrete.Discrete):
        #     raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        # if not isinstance(action_space, discrete.Discrete):
        #     raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.1,
            "eps": 0.01,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 1000}        # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        # print(observation)
        action = np.argmax(self.q[observation]) if np.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q
        ep_r = 0
        for t in range(config["n_iter"]):
            action= self.act(obs)
            # print action
            obs2, reward, done, _ = env.step(action)
            obs2 = obs2
            future = 0.0
            if not done:
                future = np.max(q[obs2])
            else:
                print(t)
                break
            q[obs][action] -= \
                self.config["learning_rate"] * (q[obs][action] - reward - config["discount"] * future)


            obs = obs2
            ep_r +=reward
        return ep_r


def main(env_name):
    env = gym.make(env_name)
    # env = discObs2Box_grid(env)
    agent = TabularQAgent(env.observation_space, env.action_space)
    for _ in range(5000):
        env.reset()
        print(agent.learn(env))

    pass


if __name__ == '__main__':
    main('PuddleWorldB-v0')


# env = gym.make(env_name)
# env = discObs2Box_grid(env)
# agent = TabularQAgent(env.observation_space, env.action_space)
# for _ in range(5000):
#     env.reset()
#     print(agent.learn(env))