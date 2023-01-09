import warnings
from typing import Tuple, Optional

from cv2 import resize, INTER_AREA, INTER_LINEAR
import gym
from gym import spaces
from gym.core import ObsType, ActType
import matplotlib.pyplot as plt
import numpy as np
import copy
import pdb
import os 

class GymEnv(gym.Env):

    HAS_HIDDEN_GOAL = set([])

    def __init__(self, exploring_starts: bool = True, start_position: Tuple = None, hidden_goal: bool = False, terminate_on_goal: bool = True, init_state: Tuple = None,should_render: bool = True, env_name = 'Pendulum-v1', init_kwargs = {}, goal_state = None, gym_rewards=True):
        
        kwargs = {'render_mode':'rgb_array'}
        for (k,v) in init_kwargs.items():
            kwargs[k] = v

        self.environment = gym.make(env_name, **kwargs)

        self._initial_state = None

        self.exploring_starts = exploring_starts if start_position is None else False

        self.fixed_goal = True

        if env_name in self.HAS_HIDDEN_GOAL:
            self.hidden_goal = hidden_goal
        
        self.terminate_on_goal = terminate_on_goal
        self.should_render = should_render
        

        self.action_space = self.environment.action_space
        self.goal_state = goal_state
        self.use_gym_rewards = gym_rewards

        self.current_state = None

        self._initialize_env_state(init_state)
        self._initialize_state_space()
        self._initialize_obs_space()
    
    def _initialize_state_space(self):
        
        #state represented as the internal observation_space of the environment
        self.state_space = self.environment.observation_space
    
    def _initialize_obs_space(self):
        
        env_screen_shape = self.environment.render().shape

        self.img_observation_space = spaces.Box(0.0, 1.0, env_screen_shape, dtype = np.float32)

        #access the already created factored state for making state_space
        self.factor_observation_space = copy.deepcopy(self.state_space)

        self.set_rendering(self.should_render)
    
    def _initialize_env_state(self, init_state = None):

        if init_state is not None:
            self.environment.state = init_state
            self._initial_state = init_state
        else:
            self.environment.reset()
    
    def _reset(self, init_state):
        self._initial_state = init_state
        self.current_state = init_state
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, dict]:

        super().reset(seed=seed)
        new_obs, _ = self.environment.reset(seed=seed)
        self._cached_state = None
        self._cached_render = None

        self._reset(new_obs)
        
        obs = self.get_observation(new_obs)
        info = self._get_info(new_obs)
        return obs, info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        
        self._cached_state = None
        self._cached_render = None

        if self.can_run(action):
            state, gym_reward, gym_terminated, truncated, info = self._step(action)
        
        self.current_state = state
        
        if self.goal_state is not None:
            if self.terminate_on_goal and self._check_goal(state):
                terminated = True
            else:
                terminated = False
        else:
            terminated = gym_terminated
        
        if self.use_gym_rewards:
            reward = gym_reward
        else:
            reward = 1 if terminated else 0

        obs = self.get_observation(state)
        info = self._get_info(state)

        if terminated:
            self.reset()

        return obs, reward, terminated, truncated, info
    
    def _step(self, action):
        obs, reward, terminated, truncated, info = self.environment.step(action)
        return obs, reward, terminated, truncated, info
    
    def can_run(self, action):

        if type(self.action_space) is gym.spaces.box.Box and type(action) is not np.ndarray:
            action = np.asarray([action], dtype=np.float32) if type(action) is not list else np.asarray(action, dtype=np.float32)
        elif type(self.action_space) is gym.spaces.discrete.Discrete:
            assert type(action) is int, 'For discrete action space, input needs to be an integer'
            

        #NOTE: input is either numpy array or list containing 'action' input
        return True if self.action_space.contains(action) else False
    
    def get_state(self):
        return self.current_state
    
    def set_state(self, state):

        is_valid = self._check_valid_state(state)

        assert is_valid, 'Attempted to call set_state with an invalid state'

        self.environment.state = np.array(state)

        self._initial_state = state
        self.current_state = state
    
    def _check_valid_state(self, state):
        
        is_valid = self.state_space.contains(list(state)) if state is not None else False

        return is_valid
    
    def is_valid_state(self, state):
        return self._check_valid_state(state)
    
    def get_observation(self, state=None):

        if state is None:
            state = self.get_state()

        if self.should_render:
            obs = self._render(state)
        else:
            obs = state

        return obs
    
    def _render(self, state=None):
        
        current_state = self.get_state()

        try:
            if state is not None:
                self.set_state(state)

            if (self._cached_state is None) or (state != self._cached_state).any():
                self._cached_state = state
                self._cached_render = self.environment.render()
            
            return self._cached_render

        finally:
            self.set_state(current_state)
    
    def _get_info(self, state=None):
        if state is None:
            state = self.get_state()
        
        info = {'state': state}
        return info
    
    def _check_goal(self, state=None):
        if state is None:
            state = self.get_state()
        
        return self.goal_state(state)
    

    def set_rendering(self, enabled=True):

        self.should_render = enabled

        if self.should_render:
            self.observation_space = self.img_observation_space
        else:
            self.observation_space = self.factor_observation_space
    
    def plot(self, ob=None, blocking=True, save=False, filename=None):

        assert self.should_render is True

        if ob is None:
            ob = self.get_observation()
        
        plt.imshow(ob)
        plt.xticks([])
        plt.yticks([])

        if blocking:
            plt.show()
        
        if save:

            if filename is None:
                raise Exception('Error in plot(): no filename provided but save = True')
            plt.savefig(filename)
        
        plt.cla()
        plt.clf()
        
    

def test_file(env_name, valid_actions, invalid_actions, init_kwargs={}, goal_state=None):

    #creating the environment
    test_env = GymEnv(env_name=env_name,init_kwargs=init_kwargs, goal_state=goal_state)

    

    print('State Space: ', test_env.state_space)
    print('Action Space: ', test_env.action_space)
    print('-------------------------------------')
    print('Factored Obs Space: ', test_env.factor_observation_space)
    print('Image Obs Space: ', test_env.img_observation_space)
    print('Observation Space: {} | Should Render: {}'.format(test_env.observation_space, test_env.should_render))

    #resetting the environment
    try:
        test_env.reset()
    except Exception as e:
        print('Error: {}'.format(e))
    
    for x in valid_actions:
        assert test_env.can_run(x) is True
    
    for x in invalid_actions:
        assert test_env.can_run(x) is False

    
    curr_state = test_env.get_state()
    print('Current State (get_state): ', curr_state)

    current_obs = test_env.get_observation()
    print('Current Observation: type: {} | shape: {} | values: {}'.format(type(current_obs), current_obs.shape, current_obs))

    new_action = test_env.action_space.sample()
    print('Sampled Action: {}'.format(new_action))

    if not os.path.isdir(os.path.relpath('./test/{}'.format(env_name))):
        os.mkdir('./test/{}'.format(env_name))

    
    test_env.plot(blocking=False, save=True, filename='./test/{}/old_obs.png'.format(env_name))

    
    obs, reward, terminated, truncated, info = test_env.step(new_action)
    print('New Post-Action State (get_state): ', test_env.get_state())

    test_env.plot(blocking=False, save=True, filename='./test/{}/action_applied.png'.format(env_name))

    new_action = test_env.action_space.sample()
    print('New Sampled Action: {}'.format(new_action))

    for i in range(10):
        obs, reward, terminated, truncated, info = test_env.step(new_action)

        if i == 3 or i==5 or i==7:
            test_env.plot(blocking=False, save=True, filename='./test/{}/step_{}.png'.format(env_name, i))

    print('New State (get_state): ', test_env.get_state())

    print('New Observation: type: {} | shape: {} | values: {}'.format(type(obs), obs.shape, obs))

    assert test_env.is_valid_state(curr_state) is True
    assert test_env.is_valid_state(None) is False

    test_env.plot(blocking=False, save=True, filename='./test/{}/final_obs.png'.format(env_name))

    test_env.set_state(curr_state)
    print('Reset State (get_state): ', test_env.get_state())

    test_env.plot(current_obs, blocking=False, save=True, filename='./test/{}/old_obs_arg.png'.format(env_name))



should_test = True
if __name__=='__main__':

    if should_test:
        pdb.set_trace()
        test_file('Pendulum-v1', init_kwargs={'g':9.81}, goal_state = lambda x: x[0]==0.0 and x[1]==0.0 and x[2]==0.0, valid_actions = [0.000001, 1.8234], invalid_actions = [-10.01, 2.000001])

        test_file('CartPole-v1', valid_actions = [0,1], invalid_actions = [-1,5])

        test_file('MountainCar-v0', goal_state = lambda x: x[0]>=0.5, valid_actions = [2,0,1], invalid_actions = [-1, 3, 4, 5])

        test_file('Acrobot-v1', goal_state = lambda x: -x[0]-(x[0]*x[2]-x[1]*x[3]) > 1.0, valid_actions = [2,0,1], invalid_actions = [-1, -4, 3,5])

        test_file('BipedalWalker-v3', init_kwargs={'hardcore':False}, valid_actions = [[-0.99,+0.9,-0.5,+0.4]], invalid_actions = [[-1.1,0.5,-0.9,0.8], [1.1, 2.00001,5,-1.00001]])


    




