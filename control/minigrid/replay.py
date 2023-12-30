import torch
from collections import deque
import random
import numpy as np

class expReplay():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=100000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward, done):
		state = np.moveaxis(obs.__array__(), 3, 1)
		state = torch.tensor(state.reshape(-1, *state.shape[2:]), dtype=torch.float)
		next_state = np.moveaxis(next_obs.__array__(), 3, 1)
		next_state = torch.tensor(next_state.reshape(-1, *next_state.shape[2:]), dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)
		done = torch.tensor([float(done)])

		self.memory.append((state, action, next_state, reward, done))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward, done = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device), done.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()

class expReplay_v2():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=100000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward, done, val_p):
		state = np.moveaxis(obs.__array__(), 3, 1)
		state = torch.tensor(state.reshape(-1, *state.shape[2:]), dtype=torch.float)
		next_state = np.moveaxis(next_obs.__array__(), 3, 1)
		next_state = torch.tensor(next_state.reshape(-1, *next_state.shape[2:]), dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)
		done = torch.tensor([float(done)])
		val_p = torch.tensor([float(val_p)])

		self.memory.append((state, action, next_state, reward, done, val_p))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward, done, val_p = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device), done.to(self.device), val_p.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()