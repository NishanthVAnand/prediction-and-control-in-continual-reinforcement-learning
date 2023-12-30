import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 32
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=256)
		self.output = nn.Linear(in_features=256, out_features=num_actions)

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
		return self.output(x)

class CNN_three_heads(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 32
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=256)
		self.output_1 = nn.Linear(in_features=256, out_features=num_actions)
		self.output_2 = nn.Linear(in_features=256, out_features=num_actions)
		self.output_3 = nn.Linear(in_features=256, out_features=num_actions)

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
		return self.output_1(x), self.output_2(x), self.output_3(x)

class CNN_half(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
		self.output = nn.Linear(in_features=128, out_features=num_actions)

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
		return self.output(x)