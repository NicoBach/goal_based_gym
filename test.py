import gym
import mujoco_py
#from envs import human_v0
#import highway_env


env = gym.make('ant_her-v2')
# env = gym.make('FetchPickAndPlace-v1')
# env = gym.make('Ant-v2')
# env = gym.make('human_stand-v0')
env.reset()

for _ in range(1000):
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        env.step(action)
    env.reset()
env.close()