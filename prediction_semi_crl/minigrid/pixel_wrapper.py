import gym
from gym.wrappers import *
from gym.spaces import *
import numpy as np
import cv2
from collections import deque

class GridImgObservation(gym.core.Wrapper):
	def __init__(self, env, PPC=6):
		super().__init__(env)
		self.PPC = PPC
		self.arr = np.zeros((3, (self.env.n+2)*self.PPC+self.env.n+1, (self.env.n+2)*self.PPC+self.env.n+1), dtype=np.float32)
		
		# surrounding walls
		self.arr[:, :self.PPC, :] = np.broadcast_to(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis], (3, self.PPC, (self.env.n+2)*self.PPC+self.env.n+1))
		self.arr[:, :, :self.PPC] = np.broadcast_to(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis], (3, (self.env.n+2)*self.PPC+self.env.n+1, self.PPC))
		self.arr[:, -self.PPC:, :] = np.broadcast_to(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis], (3, self.PPC, (self.env.n+2)*self.PPC+self.env.n+1))
		self.arr[:, :, -self.PPC:] = np.broadcast_to(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis], (3, (self.env.n+2)*self.PPC+self.env.n+1, self.PPC))

		# gray borders
		for	idx, ix in enumerate(range(self.PPC, (self.env.n+2)*self.PPC, self.PPC)):
			self.arr[:, ix+idx, :] = np.array([0.5, 0.5, 0.5])[:, None]
			self.arr[:, :, ix+idx] = np.array([0.5, 0.5, 0.5])[:, None]

		# green goal
		for cells in self.env.goal:
			cells = (cells[0]+1, cells[1]+1)
			self.arr[:, cells[0]*self.PPC+cells[0]:(cells[0]+1)*self.PPC+cells[0], cells[1]*self.PPC+cells[1]:(cells[1]+1)*self.PPC+cells[1]] = np.broadcast_to(np.array([0., 1., 0.])[:, np.newaxis, np.newaxis], (3, self.PPC, self.PPC))
			
		self.observation_space = Box(low=0, high=1, shape=(3, (self.env.n+2)*self.PPC+self.env.n+1, (self.env.n+2)*self.PPC+self.env.n+1), dtype=np.uint8)

	def observation(self, state):
		agent = (state[0]+1, state[1]+1)
		arr = self.arr.copy()
		arr[:, agent[0]*self.PPC+agent[0]:(agent[0]+1)*self.PPC+agent[0], agent[1]*self.PPC+agent[1]:(agent[1]+1)*self.PPC+agent[1]] = np.broadcast_to(np.array([1., 0., 0.])[:, np.newaxis, np.newaxis], (3, self.PPC, self.PPC))
		return np.moveaxis(arr, 0, 2)

	def step(self, action):
		state, rew, done, info = self.env.step(action)
		obs = self.observation(state)
		return obs, rew, done, info

	def reset(self, **kwargs):
		state = self.env.reset(**kwargs)
		obs = self.observation(state)
		return obs

class MCObsWrapper(gym.ObservationWrapper):
	def __init__(self, env, size=(48,48)):
		super().__init__(env)
		self.size = size
		self.observation_space = Box(low=np.min(env.observation_space.low), high=np.min(env.observation_space.low), shape=self.size)

	def observation(self, state):
		broad_state = np.tile(state[:,None,None], (1,) + self.size)
		return broad_state

	def reset(self, **kwargs):
		state = self.env.reset(**kwargs)
		obs = self.observation(state)
		return obs

	def step(self, action):
		state, rew, done, info = self.env.step(action)
		obs = self.observation(state)
		return obs, rew, done, info

	def MCObs(self, state):
		obs = self.reset(state=state)
		return obs

class PixelObsProcess(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = Box(low=0, high=255, shape=self.env.observation_space.sample()['pixels'].shape)

	def observation(self, state):
		pixel_state = state['pixels']
		return pixel_state

	def step(self, action):
		state, rew, done, info = self.env.step(action)
		obs = self.observation(state)
		return obs, rew, done, state['state']

	def reset(self, **kwargs):
		state = self.env.reset(**kwargs)
		obs = self.observation(state)
		return obs, state['state']