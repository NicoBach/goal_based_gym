import numpy as np
from gym import utils
from gym.envs.registration import register
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import rotations

PATH_ANT = '/home/nico/PycharmProjects/standingv2/standing/envs/assets/ant.xml'
DEFAULT_SIZE = 500


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.ones((7), dtype='float64')
        self.start_point = np.ones((7), dtype='float64')
        self.distance_threshold = 0.13
        self.rotation_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, PATH_ANT, 2)
        utils.EzPickle.__init__(self)
        self.start_point = self._get_achieved_goal().copy()
        self.goal = self._sample_goal(self.start_point)


    def step(self, a):
        # xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #           and state[2] >= 0.2 and state[2] <= 1.0
        done = False
        ob = self._get_obs()
        info = {
            'is_success': self._is_success(ob['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(ob['achieved_goal'], self.goal, info['is_success'])

        return ob, reward, done, info

    def _get_obs(self):
        achieved_goal = self._get_achieved_goal().ravel()
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        return {'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal,
                }

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self._is_success(achieved_goal,goal)
        return (d - 1.)

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        a = (d_pos < self.distance_threshold).astype(np.float32)
        b = (d_rot < self.rotation_threshold).astype(np.float32)
        both = a * b
        return both

    def _goal_distance(self, goal_a, goal):
        assert goal_a.shape == goal.shape

        delta_pos = goal_a[...,:2] - (goal[...,:2] + self.start_point[...,:2])
        d_pos = np.linalg.norm(delta_pos,axis=-1)
        # assert goal_a.shape[-1] == 2
        quat_a, quat_b = goal_a[..., 3:], goal[..., 3:]

        # ignore x and y dimension
        euler_a = rotations.quat2euler(quat_a)
        euler_b = rotations.quat2euler(quat_b)
        euler_a[0] = euler_b[0]
        euler_a[1] = euler_b[1]

        quat_a = rotations.euler2quat(euler_a)

        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff

        return d_pos, d_rot

    def _get_achieved_goal(self):
        # Object position
        pos = self.sim.data.get_joint_qpos("root")
        return pos

    def _sample_goal(self, sp):
        xy =  np.random.normal(self.sim.data.get_joint_qpos("root")[:2].copy(), 0.5) - sp[:2]
        xyz = np.append(xy, 0)

        quat = self.sim.data.get_joint_qpos("root")[3:].copy()
        e = rotations.quat2euler(quat)
        e[-1] = e[-1] + np.random.uniform(-np.pi, np.pi)
        quat = rotations.euler2quat(e)

        return np.append(xyz, quat)
        # g[3] =+0.5
        # g[6] =+0.5
        # return g

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.01, high=.01)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .01
        self.set_state(qpos, qvel)
        self.goal = self._sample_goal(self.start_point)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = np.append(self.goal[:2],0) - sites_offset[0]
        # print(self.sim.model.site_pos)
        # print(self.sim.data.get_body_xpos("torso"))
        self.sim.model.site_size[site_id] = self.distance_threshold
        self.sim.forward()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            camera_id = None
            camera_name = 'track'
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()


