import gym
from gym import Wrapper
from gym.wrappers import LazyFrames
from gym.spaces import Box

from collections import deque
import numpy as np

class FrameStack(Wrapper):
	"""
	Observation wrapper that stacks the observations in a rolling manner.
	For example, if the number of stacks is 4, then the returned observation contains
	the most recent 4 observations. For environment 'Pendulum-v0', the original observation
	is an array with shape [3], so if we stack 4 observations, the processed observation
	has shape [4, 3].
	note::
		To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
	note::
		The observation space must be `Box` type. If one uses `Dict`
		as observation space, it should apply `FlattenDictWrapper` at first.
	Example::
		>>> import gym
		>>> env = gym.make('PongNoFrameskip-v0')
		>>> env = FrameStack(env, 4)
		>>> env.observation_space
		Box(4, 210, 160, 3)
	Args:
		env (Env): environment object
		num_stack (int): number of stacks
		lz4_compress (bool): use lz4 to compress the frames internally
	"""
	def __init__(self, env, num_stack, lz4_compress=False):
		super(FrameStack, self).__init__(env)
		self.num_stack = num_stack
		self.lz4_compress = lz4_compress

		self.feature_frames = deque(maxlen=num_stack)

		feature_low = np.repeat(self.feature_space.low[np.newaxis, ...], num_stack, axis=0)
		feature_high = np.repeat(self.feature_space.high[np.newaxis, ...], num_stack, axis=0)
		self.feature_space = Box(low=feature_low, high=feature_high, dtype=self.feature_space.dtype)

	def _get_feature(self):
		assert len(self.feature_frames) == self.num_stack, (len(self.feature_frames), self.num_stack)
		return LazyFrames(list(self.feature_frames), self.lz4_compress)

	def step(self, action):
		state, reward, done, info = self.env.step(action)
		scent, observation, feature = state
		self.feature_frames.append(feature)
		return self._get_feature(), reward, done, info

	def reset(self, **kwargs):
		state = self.env.reset(**kwargs)
		scent, observation, feature = state
		[self.feature_frames.append(feature) for _ in range(self.num_stack)]
		return self._get_feature()