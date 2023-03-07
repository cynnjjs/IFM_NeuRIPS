import argparse
import os
import shutil
import time
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from utils import AverageMeter, accuracy
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import special_ortho_group
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
state = {}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

def mnist_model(layers, input_size, no_shrink=False):
    architecures = [[], [24], [96, 24], [128, 50, 24], [192, 96, 48, 24], [], [400, 300, 200, 100, 50, 24]]

    hidden_sizes = architecures[layers]
    if no_shrink:
        hidden_sizes = [24, 24, 24]

    modules = [(nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()))]
    for l in range(len(hidden_sizes)-1):
        modules.append(nn.Sequential(nn.Linear(hidden_sizes[l], hidden_sizes[l+1]), nn.ReLU()))

    model = nn.Sequential(*modules)
    if use_cuda:
        model = model.cuda()
    return model

def mnist_dataset():
    d2 = 28

    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=True, transform=transform)
    S_images = trainset.data.float() / 255.0
    S_labels = trainset.targets
    testset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=False, train=False, transform=transform)
    T_images = testset.data.float() / 255.0
    T_labels = testset.targets
    images = np.concatenate((S_images, T_images), axis=0)
    labels = np.concatenate((S_labels, T_labels), axis=0)

    num_data = math.floor(images.shape[0] / 12)
    num_train = int(num_data*0.8)
    num_val = int(num_data*0.1)
    num_test = num_data - num_train - num_val

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    # Create spurious means for each class
    mu2 = 2.5
    mu_e = np.random.normal(0, mu2, (10, d2))

    for e in range(12):
        col_lab = labels[e*num_data:(e+1)*num_data-num_test]
        A = np.random.randn(d2, d2)
        cov_e = np.dot(A, A.transpose())
        mean_e = mu_e[col_lab]
        color_noise = np.random.multivariate_normal(np.zeros(d2), cov_e, num_train+num_val)
        colors = mean_e + color_noise
        S_images = np.reshape(images[e*num_data:(e+1)*num_data-num_test], (num_train+num_val, 784))
        S_images = np.concatenate((S_images, colors), axis=-1)

        train_images = S_images[:num_train]
        train_labels = labels[e*num_data:e*num_data+num_train]
        val_images = S_images[num_train:num_train+num_val]
        val_labels = labels[e*num_data+num_train:e*num_data+num_train+num_val]

        col_lab = np.random.randint(0, 10, num_test)
        A = np.random.randn(d2, d2)
        cov_e = np.dot(A, A.transpose())
        mean_e = mu_e[col_lab]
        color_noise = np.random.multivariate_normal(np.zeros(d2), cov_e, num_test)
        colors = mean_e + color_noise

        T_images = np.reshape(images[(e+1)*num_data-num_test:(e+1)*num_data], (num_test, 784))
        T_images = np.concatenate((T_images, colors), axis=-1)

        T_labels = labels[(e+1)*num_data-num_test:(e+1)*num_data]

        train = data.TensorDataset(torch.Tensor(train_images),torch.Tensor(train_labels).long())
        val = data.TensorDataset(torch.Tensor(val_images),torch.Tensor(val_labels).long())
        test = data.TensorDataset(torch.Tensor(T_images),torch.Tensor(T_labels).long())

        train_dataloaders.append(data.DataLoader(train, batch_size=1000, shuffle=True))
        val_dataloaders.append(data.DataLoader(val, batch_size=1000, shuffle=True))
        test_dataloaders.append(data.DataLoader(test, batch_size=1000, shuffle=True))

    return train_dataloaders, val_dataloaders, test_dataloaders

def coral_penalty(x, y):
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff

def irm_penalty(logits, y):
    device = "cuda" if logits[0][0].is_cuda else "cpu"
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result

def task(args, manualSeed):
    print(args, 'Seed', manualSeed)
    np.random.seed(0)
    torch.manual_seed(manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(manualSeed)

    if args['irm']:
        penalty = irm_penalty
    else:
        penalty = coral_penalty

    d2 = 28
    d1 = 784
    train_dataloaders, val_dataloaders, test_dataloaders = mnist_dataset()

    model = mnist_model(args['layer'], d1+d2, args['no_shrink'])
    classifier = nn.Linear(24, 10)
    if use_cuda:
        classifier = classifier.cuda()
    all_params = list(classifier.parameters())
    for l in model:
        all_params += list(l.parameters())

    optimizer = optim.SGD(all_params, lr=args['lr'], momentum=0.9, weight_decay=1e-4)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    for epoch in range(400):
        adjust_learning_rate(args, optimizer, epoch)

        train_loss, train_acc = train(args, train_dataloaders, model, classifier, criterion, penalty, optimizer, epoch, use_cuda)
        val_loss, val_acc = val(args, val_dataloaders, model, classifier, criterion, penalty, epoch, use_cuda)
        test_loss, test_acc = val(args, test_dataloaders, model, classifier, criterion, penalty, epoch, use_cuda)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        if math.isnan(train_loss):
            return -1

    writer.close()
    
    return test_acc

def train(args, train_dataloaders, model, classifier, criterion, penalty, optimizer, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    pen_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in train_dataloaders])
    env_pr = int(12 / len(model))

    iters = [iter(e) for e in train_dataloaders]
    for batch_idx in range(num_batches):

        inputs_list = []
        targets_list = []
        for e in range(len(train_dataloaders)):
            try:
                inputs, targets = next(iters[e])
            except StopIteration:
                print('Data loading error')
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_list.append(targets)
            inputs_list.append(inputs)

        data_time.update(time.time() - end)

        # Supervised loss
        inputs = torch.cat(inputs_list)
        targets = torch.cat(targets_list)
        features = [inputs]
        for l in model:
            features.append(l(features[-1]))
        outputs = classifier(features[-1])
        sup_loss = criterion(outputs, targets)
        # Logging
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), outputs.size(0))
        sup_losses.update(sup_loss.data.item(), outputs.size(0))

        # Matching loss
        penalties = torch.zeros(1)
        if use_cuda:
            penalties = penalties.cuda()
        if args['plambda'] > 0:
            n_per_env = inputs_list[0].size()[0]
            features = inputs
            if args['irm']:
                for i in range(12):
                    penalties += penalty(outputs[i*n_per_env:(i+1)*n_per_env], targets[i*n_per_env:(i+1)*n_per_env])
                penalties /= 12
            else:
                for l in range(len(model)):
                    features = model[l](features)

                    if args['last'] and (l < len(model)-1):
                        continue

                    if args['all']:
                        lrange = 0
                        rrange = 12
                    else:
                        lrange = l*env_pr
                        rrange = (l+1)*env_pr

                    for i in range(lrange,rrange-1):
                        penalties += penalty(features[i*n_per_env:(i+1)*n_per_env], features[(i+1)*n_per_env:(i+2)*n_per_env])
                penalties /= (rrange-lrange-1)
        pen_losses.update(penalties.data.item(), outputs.size(0))

        loss = sup_loss + args['plambda'] * penalties

        losses.update(loss.data.item(), outputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def val(args, dataloaders, model, classifier, criterion, penalty, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    pen_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in dataloaders])
    env_pr = int(12 / len(model))

    iters = [iter(e) for e in dataloaders]
    for batch_idx in range(num_batches):
        inputs_list = []
        targets_list = []
        for e in range(len(dataloaders)):
            try:
                inputs, targets = next(iters[e])
            except StopIteration:
                print('Data loading error')
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_list.append(targets)
            inputs_list.append(inputs)

        data_time.update(time.time() - end)

        # Supervised loss
        with torch.no_grad():
            inputs = torch.cat(inputs_list)
            targets = torch.cat(targets_list)
            features = [inputs]
            for l in model:
                features.append(l(features[-1]))
            outputs = classifier(features[-1])
            sup_loss = criterion(outputs, targets)
            # Logging
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            top1.update(prec1.item(), outputs.size(0))
            sup_losses.update(sup_loss.data.item(), outputs.size(0))

        # Matching loss
        penalties = torch.zeros(1)
        if use_cuda:
            penalties = penalties.cuda()
        n_per_env = inputs_list[0].size()[0]

        if args['irm']:
            for i in range(12):
                penalties += penalty(outputs[i*n_per_env:(i+1)*n_per_env], targets[i*n_per_env:(i+1)*n_per_env])
            penalties /= 12
        else:
            with torch.no_grad():
                features = inputs
                for l in range(len(model)):
                    features = model[l](features)

                    if args['last'] and (l < len(model)-1):
                        continue

                    if args['all']:
                        lrange = 0
                        rrange = 12
                    else:
                        lrange = l*env_pr
                        rrange = (l+1)*env_pr
                    for i in range(lrange,rrange-1):
                        penalties += penalty(features[i*n_per_env:(i+1)*n_per_env], features[(i+1)*n_per_env:(i+2)*n_per_env])
                penalties /= (rrange-lrange-1)
        pen_losses.update(penalties.data.item(), outputs.size(0))

        with torch.no_grad():
            loss = sup_loss + args['plambda'] * penalties
            losses.update(loss.data.item(), outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch==0:
        state['lr'] = args['lr']
    if epoch in [300, 375]:
        state['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
