try:
	import gym
	from gym import spaces, logger
	modules_loaded = True
except:
	modules_loaded = False

import numpy as np
from skimage.util.shape import view_as_windows
import gc

import jbw
from jbw.agent import Agent
from jbw.direction import RelativeDirection
from jbw.item import *
from jbw.simulator import *
from jbw.visualizer import MapVisualizer, pi

class JBWEnv(gym.Env):
	"""
	JBW environment for OpenAI gym.
	The action space consists of four actions:
	- `0`: Move forward.
	- `1`: Move left.
	- `2`: Move right.
	- `3`: Move backward.
	The observation space consists of a dictionary:
	- `scent`: Vector with shape `[S]`, where `S` is the 
	  scent dimensionality.
	- `vision`: Matrix with shape `[2R+1, 2R+1, V]`, 
	  where `R` is the vision range and `V` is the 
	  vision/color dimensionality.
	- `moved`: Binary value indicating whether the last 
	  action resulted in the agent moving.
	"""

	def __init__(self, sim_config, reward_fn, render=False, f_type="obj"):
		"""
		Creates a new JBW environment for OpenAI gym.
		Arguments:
		sim_config(SimulatorConfig) Simulator configuration
									  to use.
		reward_fn(callable)         Function that takes the 
									  previously collected 
									  items and the current 
									  collected items as inputs
									  and returns a reward 
									  value.
		render(bool)                Boolean value indicating 
									  whether or not to support 
									  rendering the 
									  environment.
		"""
		self.sim_config = sim_config
		self._sim = None
		self._painter = None
		self._reward_fn = reward_fn
		self._render = render
		self.T = 0
		self.f_type = f_type

		# Computing shapes for the observation space.
		self.hash_dict = {(0,0,0):0, (1,0,0):1, (0,1,0):2, (0,0,1):3}
		self.shape_set = [(2,2), (3,3), (4,4)]
		self.get_features = self.feature_picker(self.f_type)

		self.reset()

		scent_dim = len(self.sim_config.items[0].scent)
		vision_dim = len(self.sim_config.items[0].color)
		vision_range = self.sim_config.vision_range
		vision_shape = [
		  2 * vision_range + 1, 
		  2 * vision_range + 1, 
		  vision_dim]

		min_float = np.finfo(np.float32).min
		max_float = np.finfo(np.float32).max
		min_scent = min_float * np.ones(scent_dim)
		max_scent = max_float * np.ones(scent_dim)
		min_vision = min_float * np.ones(vision_shape)
		max_vision = max_float * np.ones(vision_shape)

		# Observations in this environment consist of a scent 
		# vector, a vision matrix, and a binary value 
		# indicating whether the last action resulted in the 
		# agent moving.
		self.observation_space = spaces.Box(low=min_vision, high=max_vision)
		self.scent_space = spaces.Box(low=min_scent, high=max_scent) 
		self.action_space = spaces.Discrete(4)
		self.feature_space = spaces.Box(low=np.zeros(self.t_size), high=np.ones(self.t_size))

	def convert(self, vector):
		vector = np.ceil(vector)
		tuple_vec = tuple(vector)
		channel = self.hash_dict[tuple_vec]
		return channel

	def feature_picker(self, ftype="hash"):
		if ftype == "hash":
			self.t_size = 2048
			def feature_func(vision_state):
				features = []
				obs_channel = np.apply_along_axis(self.convert, 2, vision_state)
				obs_hash = obs_channel.choose(self.hash_vals)
				for s in self.shape_set:
					ids = np.bitwise_xor.reduce(view_as_windows(obs_hash, s), axis=(2,3)).flatten()%self.t_size
					features.extend(list(ids))
				feature_state = np.zeros(self.t_size)
				feature_state[list(set(features))] = 1
				return feature_state.flatten()
			return feature_func

		elif ftype == "obj":
			self.t_size = (len(self.hash_dict) - 1) * (self.sim_config.vision_range * 2 + 1)**2
			def feature_func(vision_state):
				features = []
				obs_channel = np.apply_along_axis(self.convert, 2, vision_state)
				obs_channel = obs_channel.flatten()
				features = np.zeros((obs_channel.size, len(self.hash_dict)))
				features[np.arange(obs_channel.size), obs_channel] = 1
				return features[:, 1:].flatten()
			return feature_func

	def step(self, action):
		prev_position = self._agent.position()
		prev_items = self._agent.collected_items()

		self._agent._next_action = action
		self._agent.do_next_action()

		position = self._agent.position()
		items = self._agent.collected_items()
		reward = self._reward_fn(prev_items, items, self.T)
		self.T += 1
		done = False

		self.vision_state = self._agent.vision()
		self.scent_state = self._agent.scent()
		self.feature_state = self.get_features(self.vision_state)

		return (self.vision_state, self.scent_state, self.feature_state), reward, done, {}

	def reset(self):
		"""Resets this environment to its initial state."""
		del self._sim
		gc.collect()
		self._sim = Simulator(sim_config=self.sim_config)
		self._agent = _JBWEnvAgent(self._sim)
		self.T = 0
		self.hash_vals = np.random.randint(0, np.iinfo(np.int32).max,\
				size=(1+len(self.sim_config.items),(2*self.sim_config.vision_range)+1,(2*self.sim_config.vision_range)+1))
		self.hash_vals[0,:,:] = 0
		if self._render:
			del self._painter
			self._painter = MapVisualizer(
			self._sim, self.sim_config, 
			bottom_left=(-70, -70), top_right=(70, 70))
		self.vision_state = self._agent.vision()
		self.scent_state = self._agent.scent()
		self.feature_state = self.get_features(self.vision_state)
		return (self.vision_state, self.scent_state, self.feature_state)

	def render(self, mode='matplotlib'):
		if mode == 'matplotlib' and self._render:
			self._painter.draw()
		elif not self._render:
			logger.warn('Need to pass `render=True` to support rendering.')
		else:
			logger.warn('Invalid rendering mode "%s". Only "matplotlib" is supported.')

	def close(self):
		del self._sim

	def seed(self, seed=None):
		self.sim_config.seed = seed
		self.reset()

class _JBWEnvAgent(Agent):
	"""
	Helper class for the JBW environment, that represents
	a JBW agent living in the simulator.
	"""

	def __init__(self, simulator):
		"""
		Creates a new JBW environment agent.
		Arguments: simulator(Simulator)  The simulator the agent lives in.
		"""
		super(_JBWEnvAgent, self).__init__(simulator, load_filepath=None)
		self._next_action = None

	def do_next_action(self):
		if self._next_action == 0:
			self.move(RelativeDirection.FORWARD)
		elif self._next_action == 1:
			self.move(RelativeDirection.LEFT)
		elif self._next_action == 2:
			self.move(RelativeDirection.RIGHT)
		elif self._next_action == 3:
			self.move(RelativeDirection.BACKWARD)
		else:
			logger.warn('Ignoring invalid action %d.' % self._next_action)
	
	def save(self, filepath):
		pass

	def _load(self, filepath):
		pass