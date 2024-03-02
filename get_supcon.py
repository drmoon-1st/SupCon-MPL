import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.nn import TripletMarginLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.model_selection import train_test_split

from data import DeepFakesDataset
from utils import set_seed, get_dataset, get_model, initialize,\
                 Classifier, get_imgs, SupConLoss, SupConResNet

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Deepfake MPL')

parser.add_argument('--model', default="resnet50", type=str, help='Model name to train.')
parser.add_argument('--seed', default=777, type=int, help='Seed value while training.')
parser.add_argument('--dataset', default='All', type=str, help='Dataset to use while training.')
parser.add_argument('--epochs', default=1000, type=int,help='Epochs to train.')
parser.add_argument('--lr', default=0.0001, type=int, help='Learning rate while training.')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size while training.')
parser.add_argument('--img_size', default=64, type=int, help='Img size while training.')
parser.add_argument('--repeat', default=1, type=int,help='Epoch repeat while train.')

args = parser.parse_args()

set_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = './weights/'

##############
## get imgs ##
##############
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

############
## models ##
############
model = get_model(args.model)

supcon = SupConResNet(model=model).to(device)
criterion = SupConLoss(temperature=0.1)
optimizer = torch.optim.Adam(supcon.parameters(), lr=args.lr)

##############
## datasets ##
##############
train_dataset = DeepFakesDataset(train_imgs, train_labels, args.img_size, mode='train', contrast=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

del train_dataset

train_iter = iter(train_loader)

def train(model, criterion, optimizer, train_iter, args):
    model.train()

    for epoch in range(args.epochs*args.repeat):
        img_a, img_b, targets = next(train_iter)

        img_a, img_b = (img_a / 255.0).to(device), (img_b / 255.0).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        images = torch.cat([img_a, img_b], dim=0)
        bsz = targets.shape[0]

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels=targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch :', epoch)
            print(f'supcon loss : {loss}')

    torch.save({f'model_state_dict': model.state_dict(),
            }, path + f'supcon_encoder.tar')

train(supcon, criterion, optimizer, train_iter, args)