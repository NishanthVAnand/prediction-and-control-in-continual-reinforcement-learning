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

class obj_net_two_heads(nn.Module):
	def __init__(self, num_channels_inp, num_actions):
		super().__init__()

		# Define image embedding
		self.image_conv = nn.Sequential(
			nn.Conv2d(num_channels_inp, 16, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(16, 32, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(32, 64, (2, 2)),
			nn.ReLU()
		)

		self.image_embedding_size = 64*2*2

		self.transient_layer = nn.Sequential(
			nn.Linear(self.image_embedding_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, num_actions)
		)

		self.permanent_layer = nn.Sequential(
			nn.Linear(self.image_embedding_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, num_actions)
		)

		self.apply(weight_init)

	def forward(self, inputs):
		x = self.image_conv(inputs)
		x = x.view(x.shape[0], -1)
		x1, x2 = self.permanent_layer(x), self.transient_layer(x)
		return x1, x2

class obj_net(nn.Module):
	def __init__(self, num_channels_inp, num_actions):
		super().__init__()

		# Define image embedding
		self.image_conv = nn.Sequential(
			nn.Conv2d(num_channels_inp, 16, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(16, 32, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(32, 64, (2, 2)),
			nn.ReLU()
		)

		self.image_embedding_size = 64*2*2

		self.critic = nn.Sequential(
			nn.Linear(self.image_embedding_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, num_actions)
		)

		self.apply(weight_init)

	def forward(self, inputs):
		x = self.image_conv(inputs)
		x = x.view(x.shape[0], -1)
		x = self.critic(x)
		return x