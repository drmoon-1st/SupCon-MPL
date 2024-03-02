import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import roc_auc_score 

from data import DeepFakesDataset
from utils import set_seed, get_dataset, get_model

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Deepfake MPL')

parser.add_argument('--model', default="resnet50", type=str, help='Model name to test.')
parser.add_argument('--seed', default=777, type=int, help='Seed value while testing.')
parser.add_argument('--train_dataset', required=True, type=str, help='Dataset to use while training.')
parser.add_argument('--test_dataset', required=True, type=str, help='Dataset to use while testing.')
parser.add_argument('--epochs', required=True, type=int,help='Epochs to test.')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size while testing.')
parser.add_argument('--img_size', default=64, type=int, help='Img size while testing.')
parser.add_argument('--mpl', default=False, type=bool, help='False: load pretrained model, True: load mpl model')
parser.add_argument('--cuda', default=0, type=int, help='gpu id')

args = parser.parse_args()

set_seed(args.seed)

imgs, labels = get_dataset(args.test_dataset, is_train=False, args=args)

model = get_model(args.model)

if os.path.exists(f'./weights/{args.model}_mpl_state_dict_{args.train_dataset}.tar'):
  loaded_info = torch.load(f'./weights/{args.model}_mpl_state_dict_{args.train_dataset}.tar')
  model.load_state_dict(loaded_info[f'model_state_dict'])
  print('mpl state loaded!!')
elif os.path.exists(f'./weights/{args.model}_state_dict_{args.train_dataset}.tar'):
  loaded_info = torch.load(f'./weights/{args.model}_state_dict_{args.train_dataset}.tar')
  model.load_state_dict(loaded_info[f'model_state_dict'])
  print('state loaded!!')

device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

print(f"Run on device {device}")

net = model.to(device)

labels = np.asarray(labels)

eval_dataset = DeepFakesDataset(np.asarray(imgs), labels, args.img_size, mode='test')
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                num_workers=12,
                pin_memory=True,
                shuffle=True)

del eval_dataset

criterion = nn.CrossEntropyLoss().to(device)

def evaluate(net, eval_iter):
    net.eval()

    correct =  0
    total = 0
    loss = 0

    _, aug_img, targets = next(eval_iter)
    aug_img, targets = (aug_img/255.0).to(device), targets.type(torch.LongTensor).to(device)

    outputs = net(aug_img)
    loss += criterion(outputs, targets).item()

    _, pred = outputs.max(1)
    correct += pred.eq(targets).sum().item()
    total += targets.size(0)

    roc_auc = roc_auc_score(targets.cpu().numpy(), pred.cpu().numpy())

    return correct / total, loss, roc_auc

evaluate_accs_f = []
evaluate_losses_f = []
evaluate_roc_f = []

path = './weights/'

eval_iter = iter(eval_loader)

for epoch in range(args.epochs):

  evaluate_acc, evaluate_loss, evaluate_roc = evaluate(net, eval_iter)
  evaluate_accs_f.append(evaluate_acc)
  evaluate_losses_f.append(evaluate_loss)
  evaluate_roc_f.append(evaluate_roc)

  # if (epoch + 1) % 100 == 0:
    # print(f"Epoch : {epoch}")
    # print(f"evaluate acc : {evaluate_acc * 100:.2f} / evaluate loss : {evaluate_loss} / roc auc : {evaluate_roc * 100:.2f}")

    # print('#'*10)

print(f"mean evaluate acc : {sum(evaluate_accs_f)/len(evaluate_accs_f) * 100:.2f}")
print("END!!!")