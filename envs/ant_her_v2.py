import numpy as np
from gym import utils
from gym.envs.registration import register
from gym.envs.mujoco import mujoco_env
from pynput import keyboard

PATH_ANT = '/home/nico/PycharmProjects/standingv2/standing/envs/assets/ant.xml'
DEFAULT_SIZE = 500




class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=False,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1):
        self.obj_range = 0.4
        self.goal = np.array([0, 0], dtype='float64')
        self.start_point = np.array([0, 0], dtype='float64')
        self.distance_threshold = 0.13
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.lis = keyboard.Listener(on_press=self.on_press)
        mujoco_env.MujocoEnv.__init__(self, PATH_ANT, 1)
        utils.EzPickle.__init__(self)
        self.start_point = self._get_achieved_goal().copy()
        self.goal = self._sample_goal()
        for _ in range(25):
            self.sim.step()
        self.lis.start() # start to listen on a separate thread
        # self.lis.join() # no this if main thread is polling self.keys

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        p = any(y != 0 for y in raw_contact_forces.flat)
        print(p)
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy


    def step(self, a):
        # xposbefore = self.get_body_com("torso")[0]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #           and state[2] >= 0.2 and state[2] <= 1.0
        self.do_simulation(a, self.frame_skip)

        ctrl_cost = self.control_cost(a)
        contact_cost = self.contact_cost

        healthy_reward = self.healthy_reward
        done = False
        ob = self._get_obs()
        info = {
            'is_success': self._is_success(ob['achieved_goal'], self.goal),
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,
        }
        rewards = self.compute_reward(ob['achieved_goal'], self.goal, info['is_success'])+ healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        return ob, reward, done, info

    def _get_obs(self):
        achieved_goal = self._get_achieved_goal().ravel()
        ant_pos = np.append(self.sim.data.get_body_xpos('torso')[:2], 0)
        dt = self.frame_skip * self.sim.model.opt.timestep
        ant_vel = self.sim.data.get_body_xvelp('torso') * dt
        ant_qpos = self.sim.data.qpos.flat[2:]
        ant_qvel = self.sim.data.qvel.flat
        goal_pos = np.append(self.goal + self.start_point, 0)
        contact_force = self.contact_forces.flat.copy()
        # goal_velp = self.sim.data.get_site_velp('target0')* dt
        # goal_velr = self.sim.data.get_site_velr('target0')* dt
        goal_rel_pos = goal_pos - ant_pos
        obs = np.concatenate([
           ant_pos, ant_vel,
            goal_pos, goal_rel_pos.ravel(), ant_qpos, ant_qvel,
            contact_force
            # goal_velp.ravel(),
            # goal_velr.ravel()
        ])
        # obs = np.zeros(111)
        return {'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal,
                }

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        d = self._goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _goal_distance(self, goal_a, goal):
        assert goal_a.shape == goal.shape
        assert goal_a.shape[-1] == 2
        delta_pos = goal_a - (goal + self.start_point)
        d_pos = np.linalg.norm(delta_pos, axis=-1)

        return d_pos

    def _get_achieved_goal(self):
        # Object position
        pos = self.sim.data.get_body_xpos("torso")[:2]
        # rel_pos = self.sim.data.get_body_xpos("torso")[:2] - (self.goal + self.start_point)
        return pos

    def _sample_goal(self):
        k = self.np_random.uniform(-self.obj_range, self.obj_range, size=2).copy()
        # k = np.random.normal(self.sim.data.get_body_xpos("torso")[:2], 0.25) -sp
        return k

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.01, high=.01)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .01
        self.set_state(qpos, qvel)
        self.start_point = self._get_achieved_goal().copy()
        self.goal = self._sample_goal().copy()
        for _ in range(25):
            self.sim.step()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = np.append(self.goal,0) + np.append(self.start_point,0) - sites_offset[0]
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


    def on_press(self,key):
        try: k = key.char # single-char keys
        except: k = key.name # other keys
        if key == keyboard.Key.esc: return False # stop listener
        # if k in ['1', '2', 'left', 'right']: # keys interested
        #     print('Key pressed: ' + k)
        if k == '8':
            # print('Key pressed: ' + k)
            self.goal[1] += 1e-2
            # print(self.goal)
        if k == '2':
           # print('Key pressed: ' + k)
           self.goal[1] -=1e-2
           # print(self.goal)
        if k == '4':
            # print('Key pressed: ' + k)
            self.goal[0] -= 1e-2
            # print(self.goal)

        if k == '6':
            # print('Key pressed: ' + k)
            self.goal[0] += 1e-2
            # print(self.goal)

            # self.keys.append(k) # store it in global-like variable
            # return False # remove this if want more keys

# register(
#     id='ant_stand-v2',
#     entry_point='envs:AntEnv',
#     max_episode_steps=100
# )
