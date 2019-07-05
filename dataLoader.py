import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore', category = UserWarning)

class DrivingDataset(Dataset):
	"""driving dataset."""

	def __init__(self, root_dir, txt_file, training = True, transform = None):
		"""
		Args:
			txt_file(string)
			root_file(string)
			training(boolean) : True for train and False for test
			transform (callable, optional): Optional transform to be applied on a sample
		"""
		self.root_dir = root_dir
		self.txt_file = txt_file
		self.transform = transform
		self.img_name = []
		self.steering_angle = []
		with open(os.path.join(root_dir, txt_file)) as f:
			for line in f:
				self.img_name.append(line.split()[0])
				# Converting steering angle which we need to predict from radians
        		# to degrees for fast computation
				self.steering_angle.append(float(line.split()[1]) * np.pi / 180)
		# 80% for train
		# 20% for test
		if(training):
			self.img_name = self.img_name[:int(len(self.img_name) * 0.8)]
			self.steering_angle = self.steering_angle[:int(len(self.steering_angle) * 0.8)]
		else:
			self.img_name = self.img_name[-int(len(self.img_name) * 0.2) - 1:]
			self.steering_angle = self.steering_angle[-int(len(self.steering_angle) * 0.2) - 1:]

	def __len__(self):
		return len(self.steering_angle)

	def __getitem__(self, idx):
		image = cv2.imread(os.path.join(self.root_dir,self.img_name[idx]))[-150:, :, :]
		image = cv2.resize(image, (200, 66))
		new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if self.transform is not None:
			new_img = self.transform(new_img)

		return new_img, torch.tensor([self.steering_angle[idx]])

'''
class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, steering_angle = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img, steering_angle

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()

class Normalize(object):

	def __call__(self, sample):
		image, steering_angle = sample
		image = (image - 0.5) / 0.5
		#tsfm = transforms.Normalize((0.5, 0.5, 0.5) , (0.5, 0.5, 0.5))
		#image = tsfm(image)
		return image, steering_angle
'''

# For test
# batch_size = 4
# num_workers = 4
# shuffle = False
# root_dir = '/home/haojieyu/Dave2/Self-Driving-Car-master/driving_dataset'
# txt_file = 'data.txt'
# transforms_composed = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#                             ])
# driving_dataset = DrivingDataset('/home/haojieyu/Dave2/Self-Driving-Car-master/driving_dataset', 'data.txt', training = False, transform = transforms_composed)
# driving_dataloader = DataLoader(driving_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

# def show_driving_dataloader_batch(sample_batched):
# 	images_batched, steering_angles_batched = sample_batched
# 	grid = utils.make_grid(images_batched * 0.5 + 0.5)
# 	plt.imshow(grid.numpy().transpose((1, 2, 0)))

# if(__name__ == '__main__'):
# 	for i_batched, sample_batched in enumerate(driving_dataloader):
# 		images_batched, steering_angles_batched = sample_batched
# 		print(images_batched)
# 		print(images_batched[0].shape)
# 		print(steering_angles_batched * 180 / np.pi)
# 		if i_batched == 5:
# 			plt.figure()
# 			show_driving_dataloader_batch(sample_batched)
# 			plt.show()
# 			break