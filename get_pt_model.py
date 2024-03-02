import os
import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.model_selection import train_test_split

from data import DeepFakesDataset
from utils import set_seed, get_dataset, get_model, get_imgs

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Deepfake MPL')

parser.add_argument('--model', default="resnet50", type=str, help='Model name to train.')
parser.add_argument('--seed', default=777, type=int, help='Seed value while training.')
parser.add_argument('--dataset', default='Sin', type=str, help='Dataset to use while training.')
parser.add_argument('--epochs', default=1000, type=int,help='Epochs to train.')
parser.add_argument('--lr', default=0.0001, type=int, help='Learning rate while training.')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size while training.')
parser.add_argument('--img_size', default=64, type=int, help='Img size while training.')
parser.add_argument('--test_size', default=0.2, type=float, help='Test size while training. (0 < test_size < 1)')
parser.add_argument('--repeat', default=1, type=int,help='Epoch repeat while train.')

args = parser.parse_args()

set_seed(args.seed)

model = get_model(args.model)

try:
    imgs_f, imgs_r, labels_f, labels_r = get_dataset(args.dataset, args=args)

    try:
        imgs_f, labels_f = get_imgs(imgs_f, labels_f, args)
        imgs_r, labels_r = get_imgs(imgs_r, labels_r, args)
    except:
        pass

    train_imgs = np.concatenate((imgs_f, imgs_r), axis=0)
    train_labels = np.concatenate((labels_f, labels_r), axis=0)
except:
    train_imgs, train_labels = get_dataset(args.dataset, args=args)

print('length of train imgs', len(train_imgs), len(train_labels))

print(len(train_labels)/args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Run on device {device}")

net = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

train_dataset = DeepFakesDataset(train_imgs, train_labels, args.img_size, mode='train')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

del train_dataset

train_iter = iter(train_loader)

def train(net, optimizer, train_iter):
    net.train()

    correct = 0
    total = 0
    losses = []

    _, aug_img, targets = next(train_iter)

    # aug_img = np.transpose(aug_img, (0, 3, 1, 2))
    aug_img, targets = (aug_img/255.0).to(device), targets.type(torch.LongTensor).to(device)

    outputs = net(aug_img)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = outputs.max(1)
    correct += pred.eq(targets).sum().item()
    total += targets.size(0)

    losses.append(loss.item())

    return correct / total, losses

train_accs_f = []
train_losses_f = []

path = './weights/'

for epoch in range(args.epochs*args.repeat):
    train_acc, train_loss = train(net, optimizer, train_iter)

    train_l = sum(train_loss)

    train_accs_f.append(train_acc)
    train_losses_f.append(train_l)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch : {epoch}")
        print(f"Train acc : {train_acc * 100:.2f} / Train loss : {train_l}")
        print()

        print('#'*10)

torch.save({f'model_state_dict': net.state_dict(),
            }, path + f'{args.model}_state_dict_{args.dataset}.tar')

print("END!!!")