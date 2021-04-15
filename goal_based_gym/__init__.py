from gym.envs.registration import register

register(
         id='ant_goal-v1',
         entry_point='goal_based_gym.envs.ant_goal_based_v1:AntGoalEnv',
         max_episode_steps=200
)

register(
         id='ant_goal-v2',
         entry_point='goal_based_gym.envs.ant_goal_based_v2:AntGoalEnv',
         max_episode_steps=100
)

register(
         id='human_her-v0',
         entry_point='goal_based_gym.envs.humanoid_goal_based_v0:HumanoidGoalEnv',
         max_episode_steps=200
)
