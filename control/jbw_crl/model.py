import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m, noise=0.1):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, 0, 0.01)
		if m.bias is not None:
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d):
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.0)

class NN_FA(nn.Module):
	def __init__(self, num_features, num_actions):
		super().__init__()

		# Define image embedding
		self.feat_layer = nn.Sequential(
			nn.Linear(num_features, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU()
		)

		self.image_embedding_size = 128

		self.critic = nn.Sequential(
			nn.Linear(self.image_embedding_size, num_actions)
		)

		self.apply(weight_init)

	def forward(self, inputs):
		x = self.feat_layer(inputs)
		x = self.critic(x)
		return x

class NN_FA_half(nn.Module):
	def __init__(self, num_features, num_actions):
		super().__init__()

		# Define image embedding
		self.feat_layer = nn.Sequential(
			nn.Linear(num_features, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU()
		)

		self.image_embedding_size = 64

		self.critic = nn.Sequential(
			nn.Linear(self.image_embedding_size, num_actions)
		)

		self.apply(weight_init)

	def forward(self, inputs):
		x = self.feat_layer(inputs)
		x = self.critic(x)
		return x

class NN_FA_two_heads(nn.Module):
	def __init__(self, num_features, num_actions):
		super().__init__()

		# Define image embedding
		self.feat_layer = nn.Sequential(
			nn.Linear(num_features, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU()
		)

		self.image_embedding_size = 128

		self.critic_1 = nn.Sequential(
			nn.Linear(self.image_embedding_size, num_actions)
		)

		self.critic_2 = nn.Sequential(
			nn.Linear(self.image_embedding_size, num_actions)
		)

		self.apply(weight_init)

	def forward(self, inputs):
		x = self.feat_layer(inputs)
		x1, x2 = self.critic_1(x), self.critic_2(x)
		return x1, x2