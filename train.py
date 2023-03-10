import argparse
import os
import time
import timm

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import MyImageNet, trans_test, trans_train
from model import VisionTransformer
from timm.models.vision_transformer import vit_base_patch16_224

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        start = time.time()
        X =X.cuda()
        y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y) / batch_size

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()
        # print("the time of one batch is--------------------------------------------------------------------:", end - start)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
         
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to("cuda"); y = y.to("cuda")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def main_ddp():

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)  # 增加local_rank
    args = parser.parse_args()
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl', init_method="env://")
    
    train_dataset = MyImageNet(types='train', transform=trans_train)
    test_dataset = MyImageNet(types="val", transform=trans_test)
    
    train_sample = DistributedSampler(train_dataset)
    test_sample = DistributedSampler(test_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, num_workers=12, sampler=train_sample)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True, num_workers=12, sampler=test_sample)
    
    model = vit_base_patch16_224()
    model = model.cuda()
    
    model = DistributedDataParallel(model, device_ids=[args.local_rank],  output_device=0,  find_unused_parameters=True)
    
    writer = SummaryWriter()
    
    learning_rate = 1e-2
    batch_size = 32
    epochs = 5
    
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

def main_single():
    train_dataset = MyImageNet(types='train', transform=trans_train)
    test_dataset = MyImageNet(types="val", transform=trans_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=True, num_workers=12)
    
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = model.cuda()
    
    writer = SummaryWriter()
    
    learning_rate = 1e-2
    batch_size = 32
    epochs = 5
    
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    
if __name__=='__main__':
    main_single()
    