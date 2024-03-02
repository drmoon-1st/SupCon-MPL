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
                 Classifier, get_imgs, SupConResNet, LinearClassifier

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
classifier = LinearClassifier().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

loaded_info = torch.load(f'./weights/supcon_encoder.tar')
supcon.load_state_dict(loaded_info[f'model_state_dict'])
print('state loaded!!')

##############
## datasets ##
##############
train_dataset = DeepFakesDataset(train_imgs, train_labels, args.img_size, mode='train')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

del train_dataset

train_iter = iter(train_loader)

def train(model, classifier, criterion, optimizer, train_iter, args):
    model.eval()
    classifier.train()

    for epoch in range(args.epochs):
        _, img, targets = next(train_iter)

        img = (img / 255.0).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        with torch.no_grad():
            features = model.encoder(img)
        output = classifier(features.detach())
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = correct = 0
        _, pred = output.max(1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

        if (epoch + 1) % 100 == 0:
            print('Epoch :', epoch)
            print(f'acc : {(correct/total*100):.2f} / loss : {loss}')

    torch.save({f'model_state_dict': classifier.state_dict(),
            }, path + f'supcon_classifier.tar')


train(supcon, classifier, criterion, optimizer, train_iter, args)