import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import albumentations
import albumentations.pytorch
import numpy as np
import math
import pandas as pd
import random
import os
import matplotlib
import argparse
import wandb

from EnD import *
from configs import *
from collections import defaultdict
import colour_mnist
import models
from tqdm import tqdm

device = torch.device('cpu')

def num_correct(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    correct = preds.eq(labels).sum()
    return correct

def train(model, dataloader, criterion, weights, optimizer, scheduler):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0
    tot_bce = 0.
    tot_abs = 0.
    model.train()

    for data, labels, color_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device), labels.to(device), color_labels.to(device)

        optimizer.zero_grad()
        with torch.enable_grad():
            outputs = model(data)
        bce, abs = criterion(outputs, labels, color_labels, weights)
        loss = bce+abs
        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size
        tot_bce += bce.item() * batch_size
        tot_abs += abs.item() * batch_size

    if scheduler is not None:
        scheduler.step()

    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss, tot_bce/num_samples, tot_abs/num_samples

def test(model, dataloader, criterion, weights):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0

    model.eval()

    for data, labels, color_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device), labels.to(device), color_labels.to(device)

        with torch.no_grad():
            outputs = model(data)
        loss = criterion(outputs, labels, color_labels, weights)

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size

    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss


def main(config):
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    train_loader, valid_loader = colour_mnist.get_biased_mnist_dataloader(
        f'{os.path.expanduser("~")}/data',
        config.batch_size,
        config.rho,
        train=True
    )

    biased_test_loader = colour_mnist.get_biased_mnist_dataloader(
        f'{os.path.expanduser("~")}/data',
        config.batch_size,
        1.0,
        train=False
    )

    unbiased_test_loader = colour_mnist.get_biased_mnist_dataloader(
        f'{os.path.expanduser("~")}/data',
        config.batch_size,
        0.1,
        train=False
    )

    print('Training debiased model')
    print('Config:', config)

    model = models.simple_convnet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    hook = Hook(model.avgpool, backward=False)

    def ce(outputs, labels, color_labels, weights):
        return F.cross_entropy(outputs, labels)

    def ce_abs(outputs, labels, color_labels, weights):
        loss = ce(outputs, labels, color_labels, weights)
        abs = abs_regu(hook, labels, color_labels, config.alpha, config.beta)
        return loss, abs

    best = defaultdict(float)

    for i in range(config.epochs):
        train_acc, train_loss, train_bce, train_abs = train(model, train_loader, ce_abs, None, optimizer, scheduler=None)
        scheduler.step()

        valid_acc, valid_loss = test(model, valid_loader, ce, None)
        biased_test_acc, biased_test_loss = test(model, biased_test_loader, ce, None)
        unbiased_test_acc, unbiased_test_loss = test(model, unbiased_test_loader, ce, None)

        print(f'Epoch {i} - Train acc: {train_acc:.4f}, train_loss: {train_loss:.4f} (bce: {train_bce:.4f} abs: {train_abs:.4f});')
        print(f'Valid acc {valid_acc:.4f}, loss: {valid_loss:.4f}')
        print(f'Biased test acc: {biased_test_acc:.4f}, loss: {biased_test_loss:.4f}')
        print(f'Unbiased test acc: {unbiased_test_acc:.4f}, loss: {unbiased_test_loss:.4f}')

        if valid_acc > best['valid_acc']:
            best = dict(
                valid_acc = valid_acc,
                biased_test_acc = biased_test_acc,
                unbiased_test_acc = unbiased_test_acc
            )

        if not config.local:
            metrics = {
                'train_acc': train_acc,
                'train_loss': train_loss,
                'train_bce': train_bce,
                'train_abs': train_abs,

                'valid_acc': valid_acc,
                'valid_loss': valid_loss,

                'biased_test_acc': biased_test_acc,
                'biased_test_loss': biased_test_loss,
                'unbiased_test_acc': unbiased_test_acc,
                'unbiased_test_loss': unbiased_test_loss,

                'best': best
            }
            wandb.log(metrics)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config}, os.path.join(wandb.run.dir, 'model.pt'))

if __name__ == '__main__':
    if not config.local:
        hyperparameters_defaults = dict(
            lr=config.lr,
            alpha=config.alpha,
            beta=config.beta,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            epochs=config.epochs,
            rho=config.rho
        )
        hyperparameters_defaults.update(config)

        tags = ['abs']
        if config.alpha == 0 and config.beta == 0:
            tags = ['baseline']
        tags.append(str(config.rho))

        wandb.init(
            config=hyperparameters_defaults,
            project='EnD-cvpr21',
            anonymous='allow',
            name=f'biased-mnist-rho{str(config.rho)}-{tags[0]}-valid',
            tags=tags,
            group=tags[0]
        )

    device = torch.device(config.device)
    main(config)
