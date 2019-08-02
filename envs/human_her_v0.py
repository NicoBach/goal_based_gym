import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
from gym import utils


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


PATH_HUMAN = '/home/nico/PycharmProjects/standingv2/standing/envs/assets/humanoid.xml'
DEFAULT_SIZE = 500

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_range = 0.1
        self.goal = np.array([0, 0, 0], dtype='float64')
        self.start_point = np.array([0,0,0], dtype='float64')
        self.distance_threshold = 0.09
        mujoco_env.MujocoEnv.__init__(self, PATH_HUMAN, 1)
        utils.EzPickle.__init__(self)
        self.start_point = self._get_achieved_goal().copy()
        self.goal = self._sample_goal()
        for _ in range(50):
            self.sim.step()

    def _get_obs(self):
        data = self.sim.data
        achieved_goal = self._get_achieved_goal()
        goal_pos = self.goal + self.start_point
        goal_rel_pos = goal_pos - achieved_goal
        obs = np.concatenate([data.qpos.flat[2:],
                              data.qvel.flat,
                              data.cinert.flat,
                              data.cvel.flat,
                              data.qfrc_actuator.flat,
                              data.cfrc_ext.flat,
                              goal_pos,goal_rel_pos
                              ])
        return {'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy(),
                }

    def step(self, a):
        # pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        # pos_after = mass_center(self.model, self.sim)
        # alive_bonus = 5.0
        # data = self.sim.data
        # lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        # #        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        done = False
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info['is_success'])

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        d = self._goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
   
    def _goal_distance(self, goal_a, goal):
        assert goal_a.shape == goal.shape
        assert goal_a.shape[-1] == 3
        delta_pos = goal_a - (goal + self.start_point)
        d_pos = np.linalg.norm(delta_pos, axis=-1)
        
        return d_pos

    def _get_achieved_goal(self):
        # Object position
        pos = self.sim.data.get_geom_xpos('head')
        return pos

    def _sample_goal(self):
        # b = np.random.normal(self.sim.data.get_geom_xpos('head')[:2].copy(), 0.1)
        # return np.append(b, self.sim.data.get_geom_xpos('head')[2].copy() - 0.12) - sp
        return np.append(self.np_random.uniform(-self.obj_range, self.obj_range, size=2).copy(), -0.135)


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        )
        self.start_point = self._get_achieved_goal().copy()
        self.goal = self._sample_goal()
        for _ in range(50):
            self.sim.step()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal + self.start_point - sites_offset[0]
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


