import time
import gym
import gym_drawobjects

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

env = gym.make("drawobjects-v0")
env.label_idx = 169 # draw panda
agent = RandomAgent(env.action_space)

reward = 0
done = False
episode_count = 3

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        print(env.action_to_human_readable(action))
        ob, reward, _, _ = env.step(action)
        print(reward)
        env.render()
        time.sleep(0.1)

env.close()
