import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from resnet import ResNet18

p = argparse.ArgumentParser()
# Choose which magnetic field parameter to predict
p.add_argument('--setting', default='default', type=str, help='what experiment to run, normal/fiveway')
p.add_argument('--mode', default='training|testing', type=str, help='what to run')

# Specify GPU to load network to and run image on
p.add_argument('--device', default='cuda:0', type=str, help='cuda GPU to run the network on')
p.add_argument('--batch_size', default=512, type=int, help='batch size')
args = p.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load datasets 
train_dataset = CIFAR100('.', train=True, download=True, transform=transform)
test_dataset = CIFAR100('.', train=False, download=True, transform=transform)

# Specify an arbitrary division of the data, first 3/5 training, next 1/5 validation.
# These ranges are sequentially assigned due to test data occuring subsequently. 
train_indices = range(int(len(train_dataset)*4/5))
val_indices = range(int(len(train_dataset)*4/5), int(len(train_dataset)))
test_indices = range(int(len(test_dataset)))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=1, pin_memory=False)
val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=1, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=False)

# Create model and initialize optimizers
net = ResNet18().to(args.device)
optimizer = optim.AdamW(net.parameters(), lr=1e-2/args.batch_size, weight_decay=1e-4, eps=1e-3)#, betas=(0.5, 0.999))
rlrop = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss()

# Training loop
epoch_len = 40000 

def load():
    options = [(x, float(x.split('_')[-1][:-4])) for x in os.listdir('./models/') if '.pth' in x and args.setting in x]
    if len(options) > 0:
        options = sorted(options, key=lambda tup: tup[1], reverse=False)
        print(options[0][0])
        model_in_path = os.path.join('./models/', options[0][0])
        load_dict = torch.load(os.path.join('./models/', options[0][0]), args.device)
        net.load_state_dict(load_dict['model'])
        print('loaded:\t{}'.format(options[0][0]))
    return net

def run_epoch(data_loader, net, optimizer, rlrop, epoch, is_train=True):
    start = time.time()
    losses = 0

    torch.set_grad_enabled(is_train)
    net = net.train() if is_train else net.eval()
    pbar = tqdm(data_loader, ncols=300)
    for i, batch in enumerate(pbar):
        # ================ preprocess in/output =====================
        im_inp = batch[0].to(args.device)
        target = batch[1].to(args.device)

        # ================== forward + losses =======================
        optimizer.zero_grad()

        pred = net(im_inp.to(args.device))

        # =============== classification target =====================
        loss = criterion(pred, target)

        if is_train:
            loss.backward()
            optimizer.step()

        # ================== logging ====================
        losses += float(loss.detach())
        step = (i + epoch * epoch_len)

        pbar.set_description(
            '{} epoch {}: itr {:<6}/ {}- {}- iml {:.4f}- aiml {:.4f}- dt {:.4f}'
          .format('TRAIN' if is_train else 'VAL  ', 
                  epoch, i * data_loader.batch_size, len(data_loader) * data_loader.batch_size, # steps
                  args.setting, loss / data_loader.batch_size, losses / (i+1), # print batch loss and avg loss
                  time.time() - start)) # batch time

        # ================== termination ====================
        if i > (epoch_len / data_loader.batch_size): break

    avg_loss = losses / (i+1)
    if not is_train:
        rlrop.step(avg_loss)
    return avg_loss

if 'training' in args.mode:
    train_losses, val_losses = [], []
    min_loss = sys.maxsize
    failed_epochs = 0
    net = load()

    for epoch in range(100):
        train_losses.append(run_epoch(train_loader, net, optimizer, rlrop, epoch, is_train=True))
        val_losses.append(run_epoch(val_loader, net, optimizer, rlrop, epoch, is_train=False))

        if val_losses[-1] < min_loss:
            min_loss = val_losses[-1]
            failed_epochs = 0
            model_out_path = './models/' + '_'.join([args.setting, str(epoch), str(float(min_loss))]) + '.pth'
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_out_path)
            print('saved model..\t\t\t {}'.format(model_out_path))
        else:
            net = load()
            failed_epochs += 1
            print('--> loss failed to decrease {} epochs..\t\t\tthreshold is {}, {} all..{}'.format(failed_epochs, 6, val_losses, min_loss))
            if failed_epochs > 4: break

if 'testing' in args.mode:
    net = load()
    torch.set_grad_enabled(False)
    net = net.eval()
    pbar = tqdm(test_loader, ncols=300)
    correct = 0
    total = 0

    for i, batch in enumerate(pbar):
        # ================ preprocess in/output =====================
        im_inp = batch[0].to(args.device)
        target = batch[1].to(args.device)
        
        pred = net(im_inp.to(args.device))

        values, indices = pred.max(dim=1)

        correct += (target == indices).sum()
        total += target.shape[0]

    print(f'Correct: {correct}')
    print(f'Total: {total}')
    print(f'%: {100.0 * (correct / (total * 1.0))}')
