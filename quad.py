import types

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self, num_inputs, num_hidden, use_tanh=True):
        super(MADE, self).__init__()

        self.use_tanh = use_tanh

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 3, num_inputs, mask_type='output')

        self.main = nn.Sequential(
            nn.MaskedLinear(num_inputs, num_hidden, input_mask), nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask), nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_inputs * 3, output_mask))

    def forward(self, inputs, mode='direct', eps=1e-8):
        if mode == 'direct':
            m, a, l = self.main(inputs).chunk(3, 1)
            if self.use_tanh:
                a = torch.tanh(a)
            if torch.min(torch.abs(l)) > eps:
                return self._quad(inputs, m, a, l, mode=mode)
            else:
                return self._linear(inputs, m, a, mode=mode)
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                m, a, l = self.main(x).chunk(3, 1)
                if self.use_tanh:
                    a = torch.tanh(a)
                J = torch.zeros(inputs.size(0),1)
                if torch.min(torch.abs(l)) > eps:
                    x_i, J_i = self._quad(inputs[:, i_col], m[:, i_col], a[:, i_col], l[:, i_col], mode=mode)
                else:
                    x_i, J_i = self._linear(inputs[:, i_col], m[:, i_col], a[:, i_col], mode=mode)
                print(J.size(), J_i.size())
                x[:, i_col] = x_i
                J += J_i.unsqueeze(1)
            return x, J

    def _quad(self, inputs, m, a, l, mode="direct"):
        if mode == 'direct':
            s = torch.exp(a)
            det = torch.sqrt(s**2 - l*(inputs - m))
            u = s + det
            J = torch.sum(torch.log(torch.abs(l)) - math.log(2) - torch.log(det), -1, keepdim=True)
            return u, J
        else:
            s = torch.exp(a)
            det = torch.sqrt(s ** 2 - l * (inputs - m))
            J = torch.log(torch.abs(l)) - math.log(2) - torch.log(det)
            x = inputs ** 2 * l - 2 * inputs * torch.exp(a) + m
            return x, J

    def _linear(self, inputs, m, a, mode="direct"):
        if mode == 'direct':
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)
        else:
            x = inputs * torch.exp(a) + m
            return x, -a

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        for module in self._modules.values():
            inputs, logdet = module(inputs)
            logdets += logdet
        return inputs, logdets

class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return (inputs[:, self.perm]), torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = ((inputs - self.batch_mean).pow(2)).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
