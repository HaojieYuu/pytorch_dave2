from model import Net
import dataLoader
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import time

'''
if th.cuda.is_available():
  # Make CuDNN Determinist
  th.backends.cudnn.deterministic = True
  th.cuda.manual_seed(seed)
'''
# Define default device, we should use the GPU (cuda) if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True

'''
reproducible
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
'''

parser = argparse.ArgumentParser(description = 'Train of Model')
parser.add_argument('-b', '--batch_size', help = 'choose batch size', type = int, default = 32)
parser.add_argument('-nw', '--num_workers', help = 'choose number of workers', type = int, default = 8)
parser.add_argument('-s', '--shuffle', help = 'decide shuffle or not', action = 'store_false')
parser.add_argument('-r', '--root_dir', help = 'choose dataset file(image)', type = str, default = '/home/haojieyu/Dave2/Self-Driving-Car-master/driving_dataset')
parser.add_argument('-t', '--training', help = 'train mode or eval mode', action = 'store_false')
parser.add_argument('-tf', '--txt_file', help = 'choose dataset file(steering angle)', type = str, default = 'data.txt')
parser.add_argument('-l', '--learning_rate', help = 'choose learning rate', type = float, default = 0.0001)
parser.add_argument('-w', '--weight_decay', help = 'choose weight decay', type = float, default = 0.0001)
parser.add_argument('-n','--n_epochs', help = 'choose number of epochs', type = int, default = 30)
parser.add_argument('-p', '--best_model_path', help = 'choose save path for best model', type = str, default = "/home/haojieyu/Pytorch_Dave2/model/best_model.pth")
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
driving_dataloader = DataLoader(driving_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

#Net
net = Net()
# net.to(device)
net.train()

#lossAndOptimizer
learning_rate = args.learning_rate
weight_decay = args.weight_decay
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)

#train
n_epochs = args.n_epochs
best_model_path = args.best_model_path

#starting train
train_history = []
n_minibatches = len(driving_dataloader)
training_start_time = time.time()
best_error = np.inf

for epoch in range(n_epochs):
    running_loss = 0.0
    count = 0
    print_every = n_minibatches // 10
    start_time = time.time()
    total_train_loss = 0

    for i_batched, sample_batched in enumerate(driving_dataloader, 0):

        images_batched, steering_angles_batched = sample_batched
        images_batched = images_batched.float()
        steering_angles_batched = steering_angles_batched.float()

        optimizer.zero_grad()

        fwd_start = time.time()
        outputs = net(images_batched)
        t_fwd = time.time() - fwd_start

        bwd_start = time.time()
        loss = criterion(outputs, steering_angles_batched)
        loss.backward()
        t_bwd = time.time() - bwd_start

        upd_start = time.time()
        optimizer.step()
        t_upd = time.time() - upd_start

        count += 1
        current_loss = loss.item()
        running_loss += current_loss 
        total_train_loss += current_loss
        print('\rEpoch: {}, {:.2%}, t_fwd: {:.2f}, t_bwd: {:.2f}, t_upd: {:.2f}'.format(epoch + 1, (i_batched + 1) / n_minibatches,
            t_fwd, t_bwd, t_upd),end = '   ')
        
        # print('\rEpoch: {}, {:.2%} Loss: {:.3f}, elapsed time: {:.2f}s'.format(epoch + 1, (i_batched + 1) / n_minibatches, running_loss/count, time.time()-start_time), end='')

        if (i_batched+1) % (print_every+1) == 0:
            print("\nEpoch: {}, {:.2%} \t train_loss: {:.6f} took: {:.2f}s".format(
            epoch + 1, (i_batched + 1) / n_minibatches, running_loss / count,
            time.time() - start_time))
            running_loss = 0.0
            count = 0
            start_time = time.time()

    train_history.append(total_train_loss / n_minibatches)
    print('\nEpoch {} finished, total_train_loss: {:.6f}'.format(epoch + 1, total_train_loss / n_minibatches))
    if total_train_loss < best_error:
        best_error = total_train_loss
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'loss': (best_error / n_minibatches),
                    'train_history': train_history,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay
                    }, best_model_path)

# print evalution of loss
print("\nTraining Finished, took {:.2f}s".format(time.time() - training_start_time))
x = np.arange(1, len(train_history) + 1)
plt.figure()
plt.plot(x, train_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolution of the training loss')
plt.show()