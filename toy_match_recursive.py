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
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
state = {}

def data_generation(num_env, d1, d2):

    num_data = 1000
    mu1 = 1
    mu2 = 10

    num_train = int(num_data*0.8)
    num_val = int(num_data*0.1)
    num_test = num_data - num_train - num_val

    mu_inv = np.ones(d1) * mu1
    cov_inv = np.eye(d1)
    s = np.eye(d1+d2)

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    for e in range(num_env):

        env_x = np.zeros((num_data, d1+d2))
        mu_e = np.random.normal(0, mu2, d2)
        A = np.random.randn(d2, d2)
        cov_e = np.dot(A, A.transpose())

        env_y = np.random.randint(2, size=num_data)

        # Train and val dataset
        for i in range(num_train + num_val):
            x_inv = np.random.multivariate_normal((env_y[i]*2-1)*mu_inv, cov_inv)
            x_e = np.random.multivariate_normal((env_y[i]*2-1)*mu_e, cov_e)
            env_x[i] = s.dot(np.concatenate((x_inv, x_e)))

        # Test dataset
        for i in range(num_train + num_val, num_data):
            x_inv = np.random.multivariate_normal((env_y[i]*2-1)*mu_inv, cov_inv)
            x_e = np.random.multivariate_normal(-(env_y[i]*2-1)*mu_e, cov_e)
            env_x[i] = s.dot(np.concatenate((x_inv, x_e)))

        train = data.TensorDataset(torch.Tensor(env_x[:num_train]),torch.Tensor(env_y[:num_train]).long())
        val = data.TensorDataset(torch.Tensor(env_x[num_train:num_train+num_val]),torch.Tensor(env_y[num_train:num_train+num_val]).long())
        test = data.TensorDataset(torch.Tensor(env_x[num_train+num_val:]),torch.Tensor(env_y[num_train+num_val:]).long())

        train_dataloaders.append(data.DataLoader(train, batch_size=100, shuffle=True))
        val_dataloaders.append(data.DataLoader(val, batch_size=100, shuffle=True))
        test_dataloaders.append(data.DataLoader(test, batch_size=100, shuffle=True))

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

    all_params = []

    d1 = 5
    d2 = 32
    num_env = args['num_env']

    train_dataloaders, val_dataloaders, test_dataloaders = data_generation(num_env, d1, d2)

    model = []
    d_spu = d2
    if args['layer'] > 1:
        divisor = d_spu ** (1.0 / args['layer'])
    else:
        divisor = d_spu
    while (d_spu > 0):
        new_dim = math.ceil(d_spu / divisor)
        if len(model) == args['layer'] - 1:
            new_dim = 0
        linear_layer = nn.Linear(d1 + d_spu, d1 + new_dim, bias=False)
        if use_cuda:
            linear_layer = linear_layer.cuda()
        model.append(linear_layer)
        all_params = all_params + list(model[-1].parameters())
        d_spu = new_dim

    classifier = nn.Linear(d1, 2)
    if use_cuda:
        classifier = classifier.cuda()

    all_params = all_params + list(classifier.parameters())
    optimizer = optim.SGD(all_params, lr=args['lr'], momentum=0.9, weight_decay=0)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    # Train and val
    if args['seq']:
        # Train feature extractors
        for l in range(len(model)):
            optimizer = optim.SGD(list(model[l].parameters()), lr=args['lr'], momentum=0.9, weight_decay=0)
            for epoch in range(500):
                adjust_learning_rate(args, optimizer, epoch, head=False)
                if args['all']:
                    train_loss = train_seq(args, l, train_dataloaders, model, penalty, optimizer, epoch, use_cuda)
                    val_loss = val_seq(args, l, val_dataloaders, model, penalty, epoch, use_cuda)
                else:
                    train_loss = train_seq(args, l, train_dataloaders[l*int(num_env / args['layer']):(l+1)*int(num_env / args['layer'])], model, penalty, optimizer, epoch, use_cuda)
                    val_loss = val_seq(args, l, val_dataloaders[l*int(num_env / args['layer']):(l+1)*int(num_env / args['layer'])], model, penalty, epoch, use_cuda)
                writer.add_scalar("train_loss_l%d" % (l), train_loss, epoch)
                writer.add_scalar("val_loss_l%d" % (l), val_loss, epoch)

                if math.isnan(train_loss):
                    return -1

        m = model[0].weight.cpu().detach().numpy()
        i = 1
        while len(model) > i:
            m = np.matmul(model[i].weight.cpu().detach().numpy(), m)
            i += 1

        # Train classifier head
        optimizer = optim.SGD(list(classifier.parameters()), lr=1e-2, momentum=0.9, weight_decay=0)

        for epoch in range(150):
            adjust_learning_rate(args, optimizer, epoch, head=True)

            train_loss, train_acc = train_head(args, train_dataloaders, model, classifier, criterion, optimizer, epoch, use_cuda)
            val_loss, val_acc = val_head(args, val_dataloaders, model, classifier, criterion, epoch, use_cuda)
            test_loss, test_acc = val_head(args, test_dataloaders, model, classifier, criterion, epoch, use_cuda)

            writer.add_scalar("train_loss_head", train_loss, epoch)
            writer.add_scalar("val_loss_head", val_loss, epoch)
            writer.add_scalar("test_loss_head", test_loss, epoch)
            writer.add_scalar("train_acc_head", train_acc, epoch)
            writer.add_scalar("val_acc_head", val_acc, epoch)
            writer.add_scalar("test_acc_head", test_acc, epoch)

    else:
        for epoch in range(500):
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
    on_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in train_dataloaders])

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
        for layer in model:
            f = layer(features[-1])
            features.append(f)
        outputs = classifier(features[-1])
        sup_loss = criterion(outputs, targets)
        # Logging
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), outputs.size(0))
        sup_losses.update(sup_loss.data.item(), outputs.size(0))

        # Matching loss
        if use_cuda:
            penalties = torch.zeros(1).cuda()
        else:
            penalties = torch.zeros(1)
        n_per_env = inputs_list[0].size()[0]
        if args['coral']:
            if args['all']:
                nmb = args['num_env']
            else:
                nmb = int(args['num_env'] / args['layer'])
            for i in range(nmb-1):
                # Separate class 0 / 1
                ilist = []
                jlist = []
                for c in range(2):
                    ilist.append(torch.nonzero(targets_list[i] == c).view(-1).cpu().numpy())
                    jlist.append(torch.nonzero(targets_list[i+1] == c).view(-1).cpu().numpy())
                    penalties += penalty(features[-1][i*n_per_env+ilist[c]], features[-1][(i+1)*n_per_env+jlist[c]])
            penalties /= nmb - 1
        elif args['irm']:
            for i in range(args['num_env']):
                penalties += penalty(outputs[i*n_per_env:(i+1)*n_per_env], targets[i*n_per_env:(i+1)*n_per_env])
            penalties /= args['num_env']
        pen_losses.update(penalties.data.item(), outputs.size(0))

        # Penalize for weight column orthogonality
        if use_cuda:
            penalty_w = torch.zeros(1).cuda()
        else:
            penalty_w = torch.zeros(1)
        for layer in model:
            wwT = torch.matmul(layer.weight, torch.t(layer.weight))
            eye_tensor = torch.eye(wwT.size()[0])
            if use_cuda:
                eye_tensor = eye_tensor.cuda()
            penalty_w += torch.norm(eye_tensor - wwT)**2
        penalty_w /= len(model)
        on_losses.update(penalty_w.data.item(), outputs.size(0))

        loss = sup_loss + args['plambda'] * penalties + args['onlambda'] * penalty_w

        losses.update(loss.data.item(), outputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def train_seq(args, l, train_dataloaders, model, penalty, optimizer, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    pen_losses = AverageMeter()
    on_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in train_dataloaders])

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

        # Get feature from layer l-1
        inputs = torch.cat(inputs_list)
        features = inputs
        for p in range(l):
            with torch.no_grad():
                features = model[p](features)
        features = model[l](features)

        # Matching loss
        if use_cuda:
            penalties = torch.zeros(1).cuda()
        else:
            penalties = torch.zeros(1)
        n_per_env = inputs_list[0].size()[0]

        if args['all']:
            nmb = args['num_env']
        else:
            nmb = int(args['num_env'] / args['layer'])
        for i in range(nmb - 1):
            # Separate class 0 / 1
            ilist = []
            jlist = []
            for c in range(2):
                ilist.append(torch.nonzero(targets_list[i] == c).view(-1).cpu().numpy())
                jlist.append(torch.nonzero(targets_list[i+1] == c).view(-1).cpu().numpy())
                penalties += penalty(features[i*n_per_env+ilist[c]], features[(i+1)*n_per_env+jlist[c]])
        penalties /= nmb - 1
        pen_losses.update(penalties.data.item(), inputs.size(0))

        # Penalize for weight column orthogonality
        if l > 0:
            with torch.no_grad():
                m = model[0].weight
            for i in range(1, l):
                with torch.no_grad():
                    m = torch.matmul(model[i].weight, m)
            m = torch.matmul(model[l].weight, m)
        else:
            m = model[0].weight

        wwT = torch.matmul(m, torch.t(m))
        eye_tensor = torch.eye(wwT.size()[0])
        if use_cuda:
            eye_tensor = eye_tensor.cuda()
        penalty_w = torch.norm(eye_tensor - wwT)**2
        on_losses.update(penalty_w.data.item(), inputs.size(0))

        loss = args['onlambda'] * penalty_w + args['plambda'] * penalties
        losses.update(loss.data.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def train_head(args, train_dataloaders, model, classifier, criterion, optimizer, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in train_dataloaders])

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
        features = inputs
        for layer in model:
            with torch.no_grad():
                features = layer(features)
        outputs = classifier(features)
        sup_loss = criterion(outputs, targets)
        # Logging
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), outputs.size(0))
        sup_losses.update(sup_loss.data.item(), outputs.size(0))

        loss = sup_loss

        losses.update(loss.data.item(), outputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def val(args, val_dataloaders, model, classifier, criterion, penalty, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    pen_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in val_dataloaders])

    iters = [iter(e) for e in val_dataloaders]
    for batch_idx in range(num_batches):

        inputs_list = []
        targets_list = []
        for e in range(len(val_dataloaders)):
            try:
                inputs, targets = next(iters[e])
            except StopIteration:
                print('Data loading error')
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_list.append(targets)
            inputs_list.append(inputs)

        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # Supervised loss
            inputs = torch.cat(inputs_list)
            targets = torch.cat(targets_list)
            features = [inputs]
            for layer in model:
                f = layer(features[-1])
                features.append(f)
            outputs = classifier(features[-1])
            sup_loss = criterion(outputs, targets)
            # Logging
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            top1.update(prec1.item(), outputs.size(0))
            sup_losses.update(sup_loss.data.item(), outputs.size(0))

        # Matching loss
        if use_cuda:
            penalties = torch.zeros(1).cuda()
        else:
            penalties = torch.zeros(1)
        n_per_env = inputs_list[0].size()[0]
        if args['coral']:
            with torch.no_grad():
                if args['all']:
                    nmb = args['num_env']
                else:
                    nmb = int(args['num_env'] / args['layer'])
                for i in range(nmb - 1):
                    # Separate class 0 / 1
                    ilist = []
                    jlist = []
                    for c in range(2):
                        ilist.append(torch.nonzero(targets_list[i] == c).view(-1).cpu().numpy())
                        jlist.append(torch.nonzero(targets_list[i+1] == c).view(-1).cpu().numpy())
                        penalties += penalty(features[-1][i*n_per_env+ilist[c]], features[-1][(i+1)*n_per_env+jlist[c]])
                penalties /= nmb - 1
        elif args['irm']:
            for i in range(args['num_env']):
                penalties += penalty(outputs[i*n_per_env:(i+1)*n_per_env], targets[i*n_per_env:(i+1)*n_per_env])
            penalties /= args['num_env']
        pen_losses.update(penalties.data.item(), outputs.size(0))

        with torch.no_grad():
            # Penalize for weight column orthogonality
            if use_cuda:
                penalty_w = torch.zeros(1).cuda()
            else:
                penalty_w = torch.zeros(1)
            for layer in model:
                wwT = torch.matmul(layer.weight, torch.t(layer.weight))
                eye_tensor = torch.eye(wwT.size()[0])
                if use_cuda:
                    eye_tensor = eye_tensor.cuda()
                penalty_w += torch.norm(eye_tensor - wwT)**2
            penalty_w /= len(model)

            loss = sup_loss + args['plambda'] * penalties + args['onlambda'] * penalty_w
            losses.update(loss.data.item(), outputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def val_seq(args, l, val_dataloaders, model, penalty, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    pen_losses = AverageMeter()
    on_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in val_dataloaders])

    iters = [iter(e) for e in val_dataloaders]
    for batch_idx in range(num_batches):

        inputs_list = []
        targets_list = []
        for e in range(len(val_dataloaders)):
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

        # Get feature from layer l-1
        inputs = torch.cat(inputs_list)
        targets = torch.cat(targets_list)
        features = inputs
        for p in range(l+1):
            with torch.no_grad():
                features = model[p](features)

        # Matching loss
        with torch.no_grad():
            if use_cuda:
                penalties = torch.zeros(1).cuda()
            else:
                penalties = torch.zeros(1)
            n_per_env = inputs_list[0].size()[0]

            if args['all']:
                nmb = args['num_env']
            else:
                nmb = int(args['num_env'] / args['layer'])
            for i in range(nmb - 1):
                # Separate class 0 / 1
                ilist = []
                jlist = []
                for c in range(2):
                    ilist.append(torch.nonzero(targets_list[i] == c).view(-1).cpu().numpy())
                    jlist.append(torch.nonzero(targets_list[i+1] == c).view(-1).cpu().numpy())
                    penalties += penalty(features[i*n_per_env+ilist[c]], features[(i+1)*n_per_env+jlist[c]])
            penalties /= nmb - 1
            pen_losses.update(penalties.data.item(), inputs.size(0))

            # Penalize for weight column orthogonality
            if l > 0:
                m = model[0].weight
                for i in range(1, l):
                    m = torch.matmul(model[i].weight, m)
                m = torch.matmul(model[l].weight, m)
            else:
                m = model[0].weight

            wwT = torch.matmul(m, torch.t(m))
            eye_tensor = torch.eye(wwT.size()[0])
            if use_cuda:
                eye_tensor = eye_tensor.cuda()
            penalty_w = torch.norm(eye_tensor - wwT)**2
            on_losses.update(penalty_w.data.item(), inputs.size(0))

            loss = args['plambda'] * penalties + args['onlambda'] * penalty_w

        losses.update(loss.data.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def val_head(args, val_dataloaders, model, classifier, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    num_batches = max([len(e) for e in val_dataloaders])

    iters = [iter(e) for e in val_dataloaders]
    for batch_idx in range(num_batches):

        inputs_list = []
        targets_list = []
        for e in range(len(val_dataloaders)):
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
        features = inputs
        for layer in model:
            with torch.no_grad():
                features = layer(features)
        with torch.no_grad():
            outputs = classifier(features)
            sup_loss = criterion(outputs, targets)
        # Logging
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), outputs.size(0))
        sup_losses.update(sup_loss.data.item(), outputs.size(0))

        loss = sup_loss
        losses.update(loss.data.item(), outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def adjust_learning_rate(args, optimizer, epoch, head=False):
    global state
    if epoch==0:
        if head:
            state['lr'] = 1e-2
        else:
            state['lr'] = args['lr']
    if head:
        s = [130, 145]
    else:
        s = [450, 480]
    if epoch in s:
        state['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
