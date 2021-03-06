
import copy
import math
import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import datasets
import flows as fnn
import general_maf as gmaf

MODEL_DIR = "trained_models/"


def build_model(num_blocks, num_inputs, num_hidden, K, M, lr, device=torch.device("cpu"), use_bn=True):
    modules = []
    for _ in range(num_blocks):
        if use_bn:
            modules += [
                gmaf.SumSqMAF(num_inputs, num_hidden, K, M),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
        else:
            modules += [
                gmaf.SumSqMAF(num_inputs, num_hidden, K, M),
                fnn.Reverse(num_inputs)
            ]
    model = fnn.FlowSequential(*modules)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.fill_(0)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    return model, optimizer

def make_datasets(train_data, val_data=None, test_data=None):
    train_dataset = _make_dataset(train_data)
    valid_dataset = _make_dataset(val_data) if val_data is not None else None
    test_dataset = _make_dataset(test_data) if test_data is not None else None
    return train_dataset, valid_dataset, test_dataset

def _make_dataset(all_data):
    if type(all_data) == list:
        data_tensors = [torch.from_numpy(data).float() for data in all_data]
        dataset = torch.utils.data.TensorDataset(*data_tensors)
    else:
        data_tensor = torch.from_numpy(all_data).float()
        dataset = torch.utils.data.TensorDataset(data_tensor)
    return dataset

def make_loaders(train_dataset, valid_dataset, test_dataset, batch_size, test_batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = None if valid_dataset is None else torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = None if test_dataset is None else torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    return train_loader, valid_loader, test_loader

def load_POWER(batch_size, test_batch_size):
    dataset = getattr(datasets, "POWER")()
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)
    train_loader, valid_loader, test_loader = make_loaders(train_dataset, valid_dataset, test_dataset,
                                                           batch_size, test_batch_size)
    return train_loader, valid_loader, test_loader


def flow_loss(u, log_jacob, use_J, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
        -1, keepdim=True)
    loss = -(log_probs + log_jacob).sum() if use_J else -(log_probs).sum()
    if size_average:
        loss /= u.size(0)
    return loss

def mse_loss(x, u, size_average=True):
    loss = (x-u).pow(2).sum(-1, keepdim=True)
    if size_average:
        loss /= u.size(0)
    return loss


def _eval(model, batch, device, mode, use_J=True, size_average=True):
    if mode == 'direct':
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)
        zhat, log_jacob = model(batch, mode='direct')
        loss = flow_loss(zhat, log_jacob, use_J=use_J, size_average=size_average)
    else:
        x, z = batch
        x = x.to(device)
        z = z.to(device)
        xhat, _ = model(z, mode='direct')
        loss = nn.MSELoss()(x, xhat)
    return loss


def train_epoch(model, optim, train_loader, epoch, device, mode, use_J, log_interval):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data_size = len(data[0]) if isinstance(data, list) else len(data)
        optim.zero_grad()
        loss = _eval(model, data, device, mode, use_J)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), epoch_loss / (log_interval)))
            epoch_loss=0

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

def validate(epoch, model, loader, device, mode, use_J=True, prefix='Validation'):
    model.eval()
    val_loss = 0

    for data in loader:
        with torch.no_grad():
            val_loss += _eval(model, data, device, mode, use_J, size_average=True).item()

    val_loss /= len(loader.dataset)
    print('\n{} set: Average loss: {:.4f}\n'.format(prefix, val_loss))

    return val_loss


def train(model, optim, train_loader, valid_loader, test_loader, epochs, device, log_interval, use_J, mode='direct'):
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(epochs):
        train_epoch(model, optim, train_loader, epoch, device, mode, use_J, log_interval)
        if valid_loader is not None:
            validation_loss = validate(epoch, model, valid_loader, device, mode, use_J=use_J)

            if epoch - best_validation_epoch >= 30:
                break

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(model)

            print('Best validation at epoch {}: Average loss: {:.4f}\n'.format(
                best_validation_epoch, best_validation_loss))

    if valid_loader is not None and test_loader is not None:
        test_loss = validate(best_validation_epoch, best_model, test_loader, device, mode, use_J=use_J, prefix='Test')
    else:
        best_model = model
        test_loss = -1
    return best_model, test_loss


#
#   NADE
#


def NADE_loss(x, w, mu, sigma):
    delta = x.expand_as(mu) - mu
    energies = torch.pow(delta, 2) * -1/2 / sigma
    max_engs,_ = energies.max(dim=2, keepdim=True)
    exps = torch.exp(energies - max_engs) * w / torch.sqrt(2 * torch.pi) / sigma
    logs = max_engs.squeeze(2) + torch.sum(exps, 2)
    return torch.sum(logs, 1)

def _eval_NADE(model, batch, device):
    if isinstance(batch, list):
        batch = batch[0]
    batch = batch.to(device)
    w, mu, sigma = model(batch)
    loss = NADE_loss(batch, w, mu, sigma)
    return loss

def train_epoch_NADE(model, optim, train_loader, epoch, device, log_interval):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data_size = len(data[0]) if isinstance(data, list) else len(data)
        optim.zero_grad()
        loss = _eval_NADE(model, data, device)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), epoch_loss / (batch_idx+1)))
            epoch_loss=0

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

def validate_NADE(epoch, model, loader, device, prefix='Validation'):
    model.eval()
    val_loss = 0

    for data in loader:
        with torch.no_grad():
            val_loss += _eval_NADE(model, data, device).item()

    val_loss /= len(loader.dataset)
    print('\n{} set: Average loss: {:.4f}\n'.format(prefix, val_loss))

    return val_loss


def train_NADE(model, optim, train_loader, valid_loader, test_loader, epochs, device, log_interval):
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(epochs):
        train_epoch_NADE(model, optim, train_loader, epoch, device, log_interval)
        validation_loss = validate(epoch, model, valid_loader, device)

        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print('Best validation at epoch {}: Average loss: {:.4f}\n'.format(
            best_validation_epoch, best_validation_loss))

    test_loss = validate(best_validation_epoch, best_model, test_loader, device, prefix='Test')
    return best_model, test_loss

#
#   Loading Models
#

def default_name(t=None, prefix="model_", suffix=".pt"):
    if t is None:
        t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    return prefix + timestamp + suffix


def load_model(name, cpu=True):
    dict = torch.load(MODEL_DIR + name, map_location='cpu') if cpu else torch.load(MODEL_DIR + name)
    model = dict['model']
    optim = dict['optim']
    args = dict['args']
    return model, optim, args