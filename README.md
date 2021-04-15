<p align="center">
  <img width="300" height="259" src="https://github.com/NicoBach/goal_based_gym/blob/master/pictures/ant_goal.gif">
</p>

Ant and Humanoid-environments made goal-based. The agent receives reward or no penalty, when it is able to move its head
into the read circle.


# Requirements

OpenAI gym and mujoco_py are required to be installed in a conda
environment. Also for training and running the pretrained models
 OpenAI Baselines is required. 

# Installation

- change into the cloned folder
- pip install -e .

# Train & run (e.g. with baselines) 

To train use from command line with conda env enabled:

```
mpirun -np 4 python -m baselines.run --alg=her --env=ant_her-v2 --num_timesteps=2e6 --relative_goals=True --save_path=~/projects/her_environments/networks/...
```
To test weights run from command line with conda env enabled:
```
python -m baselines.run --alg=her --env=ant_her-v2 --num_timesteps=0 --load_path=~/projects/her_environments/networks/... --play
```
