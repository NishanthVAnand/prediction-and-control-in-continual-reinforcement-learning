import torch
from collections import deque
import random
import numpy as np

class expReplay_NN():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=8000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward):
		state = obs.__array__().flatten()
		state = torch.tensor(state, dtype=torch.float)
		next_state = next_obs.__array__().flatten()
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)

		self.memory.append((state, action, next_state, reward))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()

class expReplay_NN_PM():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=10000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, val_p):
		state = obs.__array__().flatten()
		state = torch.tensor(state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		val_p = torch.tensor([val_p], dtype=torch.float)

		self.memory.append((state, action, val_p))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, val_p = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), val_p.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()

class expReplay_NN_Large():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=250000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward):
		state = obs.__array__().flatten()
		state = torch.tensor(state, dtype=torch.float)
		next_state = next_obs.__array__().flatten()
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)

		self.memory.append((state, action, next_state, reward))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()