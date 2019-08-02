# Requirements

OpenAI gym and mujoco_py are required to be installed in a conda
environment. Also for training and running the pretrained models
 OpenAI Baselines is required. 

# Installation

The files in the envs folder should go to the gym/gym/envs/mujoco folder.
Then it is also necessary to edit gym/gym/envs/\_\_init\_\_.py to register 
the environments with the following lines.

```

register(
    id='ant_her-v2',
    entry_point='gym.envs.mujoco.ant_her_v2:AntEnv',
    max_episode_steps=200
)

register(
    id='ant_her-v3',
    entry_point='gym.envs.mujoco.ant_her_v3:AntEnv',
    max_episode_steps=100
)

register(
    id='human_her-v0',
    entry_point='gym.envs.mujoco.human_her_v0:HumanoidEnv',
    max_episode_steps=200
)

```

Then it is also necessary to fit the path to the XML-Files used here in
ant_her_v2.py and the other files. To test the environments, run test.py.

To train use from command line with conda env enabled:

```
mpirun -np 4 python -m baselines.run --alg=her --env=ant_her-v2 --num_timesteps=2e6 --relative_goals=True --save_path=~/policies/her/name
```
To test weights run from command line with conda env enabled:
```
python -m baselines.run --alg=her --env=ant_her-v2 --num_timesteps=0 --load_path=~/policies/her/name --play
```
