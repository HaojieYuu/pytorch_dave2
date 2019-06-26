import torch.optim as optim
import torch.nn as nn

def createLossAndOptimizer(net, learning_rate = 0.001, weight_decay = 0.001):
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
	return criterion, optimizer