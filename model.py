import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 24, 5, stride = 2)
		self.conv2 = nn.Conv2d(24, 36, 5, stride = 2)
		self.conv3 = nn.Conv2d(36, 48, 5, stride = 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.fc1 = nn.Linear(64 * 1 * 18, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)
		self.dropout1 = nn.Dropout(p = 0.6)
		self.dropout2 = nn.Dropout(p = 0.6)
		self.dropout3 = nn.Dropout(p = 0.6)

	def forward(self, x):
		# x: [batch_size, C, H, W]
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = x.view(-1, 64 * 1 * 18)
		x = F.relu(self.fc1(x))
		x = self.dropout1(x)
		x = F.relu(self.fc2(x))
		x = self.dropout2(x)
		x = F.relu(self.fc3(x))
		x = self.dropout3(x)
		x = self.fc4(x)
		return x