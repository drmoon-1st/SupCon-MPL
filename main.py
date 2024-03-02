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
                get_imgs, SupConLoss, SupConResNet, LinearClassifier

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Deepfake MPL')

parser.add_argument('--model', default="resnet50", type=str, help='Model name to train.')
parser.add_argument('--seed', default=777, type=int, help='Seed value while training.')
parser.add_argument('--dataset', default='All', type=str, help='Dataset to use while training.')
parser.add_argument('--dataset_u', default='Celebdf', type=str, help='Unlabeled dataset to use while training.')
parser.add_argument('--epochs', default=1000, type=int,help='Epochs to train.')
parser.add_argument('--lr', default=0.0001, type=int, help='Learning rate while training.')
parser.add_argument('--batch_size_l', default=64*1, type=int, help='Batch size of labeled dataset while training.')
parser.add_argument('--batch_size_ul', default=64*7, type=int, help='Batch size of unlabeled data while training.')
parser.add_argument('--img_size', default=64, type=int, help='Img size while training.')
parser.add_argument('--threshhold', default=0.95, type=float, help='Threshhold of logits.')
parser.add_argument('--T', default=1, type=int, help='Temperature while training.')
parser.add_argument('--repeat', default=1, type=int,help='Epoch repeat of labeled data while training.')
parser.add_argument('--repeat_epoch', default=1, type=int,help='Epoch repeat of labeled data while finetuneing.')
parser.add_argument('--un', default=False, type=bool,help='get unlabeled image for training.')


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
    if args.un:
        train_imgs, train_labels, unlabeled_imgs, unlabeled_labels = get_dataset(args.dataset, args=args, un=args.un)
    else:
        train_imgs, train_labels = get_dataset(args.dataset, args=args)

############
## models ##
############
teacher, student = get_model(args.model, is_meta=True)

supcon_t = SupConResNet(model=teacher).to(device)
supcon_s = SupConResNet(model=student).to(device)

classifier_t = LinearClassifier().to(device)
classifier_s = LinearClassifier().to(device)

loaded_info_sup = torch.load(f'./weights/supcon_encoder.tar')
supcon_t.load_state_dict(loaded_info_sup[f'model_state_dict'])
print('supcon state loaded!!')

loaded_info_clf = torch.load(f'./weights/supcon_classifier.tar')
classifier_t.load_state_dict(loaded_info_clf[f'model_state_dict'])
print('classifier state loaded!!')

criterion_ce = nn.CrossEntropyLoss().to(device)
criterion_supcon = SupConLoss(temperature=0.1).to(device)

optimizer_t_sup = torch.optim.Adam(supcon_t.parameters(), lr=args.lr)
optimizer_t_clf = torch.optim.Adam(classifier_t.parameters(), lr=args.lr)
optimizer_s_sup = torch.optim.Adam(supcon_s.parameters(), lr=args.lr)
optimizer_s_clf = torch.optim.Adam(classifier_s.parameters(), lr=args.lr)

##############
## datasets ##
##############
train_dataset = DeepFakesDataset(train_imgs, train_labels, args.img_size, mode='train', contrast=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size_l,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

unlabeled_dataset = DeepFakesDataset(np.asarray(unlabeled_imgs), np.asarray(unlabeled_labels), args.img_size, mode='train',
                                     contrast=True)

unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size_ul,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

print('length of unlabeled imgs :', len(unlabeled_labels))

del train_dataset, unlabeled_dataset

train_iter = iter(train_loader)
unlabeled_iter = iter(unlabeled_loader)

###################
## training loop ##
###################
def train(supcon_t, supcon_s, classifier_t, classifier_s, labeled_iter, unlabeled_iter,
          criterion_ce, criterion_supcon, optimizer_t_sup, optimizer_t_clf, optimizer_s_sup,
          optimizer_s_clf, args):
    supcon_t.train();supcon_s.train();classifier_t.train();classifier_s.train()

    for epoch in range(args.epochs*args.repeat):
        img_a, img_b, targets = next(labeled_iter)
        img_a, img_b = (img_a / 255.0).to(device), (img_b / 255.0).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        try:
            img_uw, img_us, _ = next(unlabeled_iter)
        except:
            unlabeled_dataset = DeepFakesDataset(np.asarray(unlabeled_imgs), np.asarray(unlabeled_labels), args.img_size, mode='train',
                                     contrast=True)

            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size_ul,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=True)
            
            unlabeled_iter = iter(unlabeled_loader)
            img_uw, img_us, _ = next(unlabeled_iter)

        img_uw, img_us = (img_uw / 255.0).to(device), (img_us / 255.0).to(device)

        # make pseudo labels
        supcon_t.eval()

        t_images = torch.cat([img_a, img_uw, img_us], dim=0)

        with torch.no_grad():
            features = supcon_t.encoder(t_images)
        t_logits = classifier_t(features.detach())
        t_logits_l = t_logits[:args.batch_size_l]
        t_logits_uw, t_logits_us = t_logits[args.batch_size_l:].chunk(2)
        del t_logits

        total = correct = 0
        _, pred = t_logits_l.max(1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

        t_loss_l = criterion_ce(t_logits_l, targets)

        supcon_t.train()

        # s_old
        supcon_s.eval()

        with torch.no_grad():
            features = supcon_s.encoder(img_a)
        s_logit_old = classifier_s(features)
        s_loss_old = F.cross_entropy(s_logit_old.detach(), targets)

        supcon_s.train()

        # update supcon_s
        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.T, dim=-1)
        _, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)

        images = torch.cat([img_uw, img_us], dim=0)
        bsz = hard_pseudo_label.shape[0]

        features = supcon_s(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_s_sup = criterion_supcon(features, labels=hard_pseudo_label)

        optimizer_s_sup.zero_grad()
        loss_s_sup.backward()
        optimizer_s_sup.step()

        # update classifier_s
        supcon_s.eval()

        with torch.no_grad():
            features = supcon_s.encoder(img_a)
        output = classifier_s(features.detach())
        loss_s_clf = criterion_ce(output, targets)

        optimizer_s_clf.zero_grad()
        loss_s_clf.backward()
        optimizer_s_clf.step()

        # s_new
        classifier_s.eval()

        with torch.no_grad():
            features = supcon_s.encoder(img_a)
            s_logit_new = classifier_s(features)
        s_loss_new = F.cross_entropy(s_logit_new.detach(), targets)

        dot_prod = s_loss_new - s_loss_old

        supcon_s.train();classifier_s.train()

        # update supcon_t
        images = torch.cat([img_a, img_b], dim=0)
        bsz = targets.shape[0]

        features = supcon_t(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_t_sup = criterion_supcon(features, labels=targets)

        optimizer_t_sup.zero_grad()
        loss_t_sup.backward()
        optimizer_t_sup.step()

        # update classifier_t -> student feedbak
        _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
        t_loss_mpl = dot_prod * F.cross_entropy(t_logits_us, hard_pseudo_label)

        t_loss = t_loss_l + t_loss_mpl

        optimizer_t_clf.zero_grad()
        t_loss.backward()
        optimizer_t_clf.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch : {epoch}')
            print(f't_acc : {(correct/total*100):.2f} / t_loss : {t_loss}')
            print(f'supcon loss t : {loss_t_sup}')
            print(f'supcon loss s : {loss_s_sup}')
            print('############')
            print()

def finetune(supcon_s, classifier_s, datas, labels, args):
    # train supcon model
    dataset = DeepFakesDataset(datas, labels, args.img_size, mode='train', contrast=True)
    loader_sup = DataLoader(dataset, batch_size=args.batch_size_l + args.batch_size_ul,
                    num_workers=12,
                    pin_memory=True,
                    shuffle=True)
    
    del dataset
    
    sup_iter = iter(loader_sup)

    optimizer_sup = torch.optim.Adam(supcon_s.parameters(), lr=args.lr)
    supconloss = SupConLoss(temperature=0.1).to(device) 

    supcon_s.train()

    for epoch in range(args.epochs*args.repeat_epoch):
        img_a, img_b, targets = next(sup_iter)

        img_a, img_b = (img_a / 255.0).to(device), (img_b / 255.0).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        images = torch.cat([img_a, img_b], dim=0)
        bsz = targets.shape[0]

        features = supcon_s(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_s = supconloss(features, labels=targets)

        optimizer_sup.zero_grad()
        loss_s.backward()
        optimizer_sup.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch : {epoch}')
            print(f'supcon loss : {loss_s}')
            print('############')
            print()

    # train classifier
    dataset = DeepFakesDataset(datas, labels, args.img_size, mode='train')
    loader_clf = DataLoader(dataset, batch_size=args.batch_size_l + args.batch_size_ul,
                    num_workers=12,
                    pin_memory=True,
                    shuffle=True)
    
    del dataset

    clf_iter = iter(loader_clf)

    optimizer_clf = torch.optim.Adam(classifier_s.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.epochs*args.repeat_epoch):
        _, img, targets = next(clf_iter)

        img = (img / 255.0).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        with torch.no_grad():
            features = supcon_s.encoder(img)
        output = classifier_s(features.detach())
        loss_c = criterion(output, targets)

        optimizer_clf.zero_grad()
        loss_c.backward()
        optimizer_clf.step()

        total = correct = 0
        _, pred = output.max(1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch : {epoch}')
            print(f'acc : {(correct/total*100):.2f} / loss : {loss_c}')
            print('############')
            print()

    supcon_s.eval()
    classifier_s.train()

train(supcon_t, supcon_s, classifier_t, classifier_s, train_iter, unlabeled_iter,
          criterion_ce, criterion_supcon, optimizer_t_sup, optimizer_t_clf, optimizer_s_sup,
          optimizer_s_clf, args)

finetune(supcon_s, classifier_s, train_imgs, train_labels, args)

torch.save({f'model_state_dict': supcon_s.state_dict(),
            }, path + f'mpl_supcon_encoder.tar')
torch.save({f'model_state_dict': classifier_s.state_dict(),
            }, path + f'mpl_supcon_classifier.tar')

print('END!!!')