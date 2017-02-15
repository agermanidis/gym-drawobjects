![gym-drawobjects](http://i.imgur.com/tscQqFy.gif)

# gym-drawobjects: An OpenAI Gym environment for drawing objects

Defines an [OpenAI Gym](https://gym.openai.com/) environment (`drawobjects-v0`) for training agents to draw ImageNet objects on a black-and-white canvas. 

### Environment

__Action space__: 6 available actions: UP, DOWN, LEFT, RIGHT, PAINT_WHITE, PAINT_BLACK

__Observation space__: canvas pixel values

__Reward__: The Inception-V3 prediction for your label on the current canvas

### Setup

```
$ pip install .
```

### Usage

```
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
steps_per_episode = 10000

for i in range(episode_count):
    ob = env.reset()
    for j in range(steps_per_episode):
        action = agent.act(ob, reward, done)
        print(env.action_to_human_readable(action))
        ob, reward, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

env.close()
```
