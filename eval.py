import torch
import torch.nn as nn
from model import Net
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import transforms
import dataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import time
import argparse
import os
import sys

parser = argparse.ArgumentParser(description = 'Evolution of Model')
parser.add_argument('model', help = 'choose model', type = str)
parser.add_argument('-b', '--batch_size', help = 'choose batch size', type = int, default = 32)
parser.add_argument('-n', '--num_workers', help = 'choose number of workers', type = int, default = 8)
parser.add_argument('-s', '--shuffle', help = 'decide shuffle or not', action = 'store_true')
parser.add_argument('-r', '--root_dir', help = 'choose dataset file(image)', type = str, default = '/home/haojieyu/Dave2/Self-Driving-Car-master/driving_dataset')
parser.add_argument('-t', '--training', help = 'train mode or eval mode', action = 'store_true')
parser.add_argument('-tf', '--txt_file', help = 'choose dataset file(steering angle)', type = str, default = 'data.txt')
args = parser.parse_args()

#dataLoader
batch_size = args.batch_size
num_workers = args.num_workers
shuffle = args.shuffle
root_dir = args.root_dir
txt_file = args.txt_file
training = args.training
transforms_composed = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                            ])
driving_dataset = dataLoader.DrivingDataset(root_dir, txt_file, training, transforms_composed)
driving_dataloader = DataLoader(driving_dataset, batch_size = batch_size, num_workers = num_workers)

checkpoint_path = os.path.join('/home/haojieyu/Pytorch_Dave2', args.model)
steering_wheel = cv2.imread('steering_wheel.jpg', 0)
rows, cols = steering_wheel.shape[:2]

#start eval
start_time = time.time()
with torch.no_grad():
	net = Net()
	criterion = nn.MSELoss()

	checkpoint = torch.load(checkpoint_path)
	net.load_state_dict(checkpoint['model_state_dict'])
	net.eval()

	#print statistics
	print('Epoch: {}, Loss: {:.6f}'.format(checkpoint['epoch'], checkpoint['loss']))
	plt.ion()
	plt.figure(1)
	train_history = checkpoint['train_history']
	x = np.arange(1, len(train_history) + 1)
	plt.plot(x, train_history)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Evolution of the training loss')

	input('press ENTER to start testing...')
	plt.close(1)

	test_history = []
	smoothed_angle = 0
	for i_batched, sample_batched in enumerate(driving_dataloader, 0):
		images_batched, steering_angles_batched = sample_batched
		images_batched = images_batched.float()
		steering_angles_batched = steering_angles_batched.float()
		prediction = net(images_batched)
		loss = criterion(prediction, steering_angles_batched)
		test_history.append(loss)

		print('Loss: {}'.format(loss))
		prediction = prediction.squeeze().numpy()
		steering_angles_batched = steering_angles_batched.squeeze().numpy()
		images_batched = images_batched.numpy()
		images_batched = images_batched * 0.5 + 0.5 # unnormalize
		for i in range(len(steering_angles_batched)):
			cv2.imshow('frame', cv2.resize(images_batched[i].transpose((1, 2, 0)), None, fx = 3, fy = 3))

			degrees = -prediction[i] * 180 / np.pi
			smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
			M = cv2.getRotationMatrix2D((cols / 2, rows / 2), smoothed_angle, 1)
			dst = cv2.warpAffine(steering_wheel, M, (cols, rows))
			cv2.imshow('steering wheel', dst)
			if cv2.waitKey(10) & 0xFF == ord('q'): # enter q for exit
				sys.exit(0)
			print('pred: {:.6f}, actual: {:.6f}'.format(prediction[i] * 180 / np.pi,
				steering_angles_batched[i] * 180 / np.pi))
	cv2.destroyAllWindows()

	#print loss
	print('testing finished, took {:.2f}s, mean loss: {:.6f}'.format(time.time() - start_time, np.mean(test_history)))
	x = np.arange(1, len(test_history) + 1)
	plt.figure(2)
	plt.plot(x, test_history)
	plt.xlabel('N_batch')
	plt.ylabel('Loss')
	plt.title('Evolution of the testing loss')
	plt.ioff()
	plt.show()