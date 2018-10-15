import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DeepFashion, CelebA, collate_fn
from model import AttrEncoder

def train(args):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor()
	])

    # dataset and dataloader for training

    # train_dataset = DeepFashion(args.root_dir, args.img_dir, args.ann_dir, transform=transform)
    train_dataset = CelebA(args.root_dir, args.img_dir, args.ann_dir, transform=transform)
    # test_dataset = DeepFashion(args.root_dir, args.img_dir, args.ann_dir, mode='Val', transform=transform)
    test_dataset = CelebA(args.root_dir, args.img_dir, args.ann_dir, mode='Val', transform=transform)
    # trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # model, optimizer, criterion
    model = AttrEncoder(outdims=args.attrnum).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss().to(device)

    print("begin training", flush=True)
    for ep in range(args.num_epoch):
        model.train()
        for it, sample in enumerate(trainloader):
            im = sample['image'].to(device)
            t = sample['attributes'].to(device)

            optimizer.zero_grad()

            out = model.forward(im)
            loss = criterion(out, t)
            loss.backward()
            optimizer.step()
            if it % 10 == 9:
                print("{}th iter \t loss: {}".format(it+1, loss), flush=True)
            
        cnt, allcnt = eval(model, testloader, device)

        print("-" * 30)
        print("epoch [{}/{}] done | accuracy [{}/{}]".format(ep, args.num_epoch, cnt, allcnt))
        print("-" * 30, flush=True)
        torch.save(model.state_dict(), "../out/epoch_{:1d}.model".format(ep))
    print("end training")

def eval(model, dataloader, device):
    cnt = 0
    allcnt = 0
    model.eval()
    for j, sample in enumerate(dataloader):
        im = sample['image'].to(device)
        t = sample['attributes'].to(device)

        out = model.forward(im)
        bs = im.size(0)
        for i in range(bs):
            if torch.equal(im[i], out[i]):
                cnt += 1
            allcnt += 1
        if j == 10:
            break
    return cnt, allcnt

def main():
    '''
    optional arguments

    num_epoch : number of epochs to train
    attrnum : number of attributes
    learning_rate : initial learning rate of Adam optimizer
    batch_size : batch size for training
    root_dir : full path of data. should be parent for images and annotations
    img_dir : relative path of directory containing images
    ann_dir : relative path of file containing annotations of attributes 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--attrnum', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--root_dir', type=str, default='../../../local/CelebA/')
    parser.add_argument('--img_dir', type=str, default='img_align_celeba')
    parser.add_argument('--ann_dir', type=str, default='list_attr_celeba.csv')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
