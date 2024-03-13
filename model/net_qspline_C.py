# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:52:22 2020

@author: 18096
"""
from tqdm import tqdm

'''Defines the neural network, loss function and metrics'''

import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.autograd import Variable
import logging

logger = logging.getLogger('DeepAR.Net')


class Net(nn.Module):
    def __init__(self, params, device):
        '''
        We define a recurrent network that predicts the future values
        of a time-dependent variable based on past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.lstm = nn.LSTM(input_size=params.lstm_input_size,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)

        # initialize LSTM forget gate bias to be 1 as recommanded by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        # Plan C:
        self.pre_beta_0 = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.pre_gamma = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, params.num_spline)

        self.beta_0 = nn.Softplus()
        # soft-plus to make sure gamma is positive
        # self.gamma=nn.ReLU()
        self.gamma = nn.Softplus()

        self.train_window = self.params.pred_steps + self.params.pred_start

    def forward(self, train_batch, labels_batch=None):  # [256, 108, 7], [256, 108]
        batch_size = train_batch.shape[0]  # 256
        device = train_batch.device

        train_batch = train_batch.permute(1, 0, 2)  # [108, 256, 7]
        if labels_batch is not None:
            labels_batch = labels_batch.permute(1, 0)  # [108, 256]

        hidden = torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim,
                             device=device)  # [2, 256, 40]
        cell = torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim,
                           device=device)  # [2, 256, 40]

        if labels_batch is not None:  # train mode
            loss = torch.zeros(1, device=device, requires_grad=True)  # [,]
            flag = False
            for t in range(self.train_window):
                # {[256, 1], [256, 20]}, [2, 256, 40], [2, 256, 40]
                x = train_batch[t].unsqueeze_(0).clone()  # [1, 256, 7]

                _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
                # use h from all three layers to calculate mu and sigma
                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [256, 80]

                # Plan C:
                pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
                beta_0 = self.beta_0(pre_beta_0)  # [256, 1]
                pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
                gamma = self.gamma(pre_gamma)  # [256, 20]
                func_param = (beta_0, torch.squeeze(gamma))  # {[256, 1], [256, 20]}

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0:
                    # logger.info('Backward Error! Process Stop!')
                    flag = True
                    # return (loss_epoch / self.train_window, flag)
                    raise ValueError(f'Backward Error! Process Stop!')
                loss = loss + loss_fn(func_param, labels_batch[t])

                # check if loss contains NaN
                if torch.isnan(loss).sum() > 0:
                    # logger.info(f'Loss Error at Data={i} Time={t}! Process Stop!')
                    flag = True
                    # return (loss_epoch / self.train_window, flag)
                    raise ValueError(f'Loss Error! Process Stop!')

            return loss, flag
        else:  # validate or test mode
            # condition range
            test_batch = train_batch  # [108, 256, 7]
            for t in range(self.params.pred_start):
                x = test_batch[t].unsqueeze(0)  # [1, 256, 7]

                _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

            # prediction range
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.pred_steps,
                                  device=device)  # [99, 256, 12]
            for j in range(self.params.sample_times):
                for t in range(self.params.pred_steps):
                    x = test_batch[self.params.pred_start + t].unsqueeze(0)  # [1, 256, 7]

                    _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
                    # use h from all three layers to calculate mu and sigma
                    hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [256, 80]

                    # Plan C:
                    pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
                    beta_0 = self.beta_0(pre_beta_0)  # [256, 1]
                    pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
                    gamma = self.gamma(pre_gamma)  # [256, 20]

                    # pred_cdf is a uniform distribution
                    uniform = torch.distributions.uniform.Uniform(
                        torch.tensor([0.0], device=device),
                        torch.tensor([1.0], device=device))
                    pred_cdf = uniform.sample([batch_size])  # [256, 1]

                    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]
                    beta = pad(gamma, (1, 0))[:, :-1]
                    beta[:, 0] = beta_0[:, 0]
                    beta = (gamma - beta) / (2 * sigma)
                    beta = beta - pad(beta, (1, 0))[:, :-1]
                    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]

                    ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [256, 20]
                    indices = ksi < pred_cdf  # [256, 20]
                    pred = (beta_0 * pred_cdf).sum(dim=1)  # [256,]
                    pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256, 20] # Q(alpha)公式?

                    samples[j, :, t] = pred
                    # predict value at t-1 is as a covars for t,t+1,...,t+lag
                    for lag in range(self.params.lag):
                        if t < self.params.pred_steps - lag - 1:
                            test_batch[self.params.pred_start + t + 1, :, 0] = pred

            sample_mu = torch.mean(samples, dim=0)  # mean or median ? # [256, 12]
            sample_std = samples.std(dim=0)  # [256, 12]
            return samples, sample_mu, sample_std

    def plot(self, data_loader, sample=False, probability_range=0.95):
        if self.device != "cpu":
            print('Use cpu for faster speed!')

        all_data = data_loader.get_all_data()  # [15632, 17]
        data = all_data[:, :-1]  # [15632, 16]
        label = all_data[:, -1]  # [15632, 1]

        data = torch.Tensor(data).to(self.device)
        label = torch.Tensor(label).to(self.device)

        cdf_high = 1 - (1 - probability_range) / 2
        cdf_low = (1 - probability_range) / 2

        total_length = data.shape[0]  # 15632
        batch_size = 1
        device = self.device
        sample_times = self.params.sample_times if sample else 1

        test_batch = data.unsqueeze(0)  # [1, 15632, 16]
        labels_batch = label.unsqueeze(0).squeeze(-1)  # [1, 15632]

        test_batch = test_batch.permute(1, 0, 2)  # [15632, 1, 16]
        if labels_batch is not None:
            labels_batch = labels_batch.permute(1, 0)  # [15632, 1]

        hidden = torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim, device=device,
                             requires_grad=False)  # [2, 256, 40]
        cell = torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim, device=device,
                           requires_grad=False)  # [2, 256, 40]

        # condition range: init hidden & cell value
        for t in range(self.params.pred_start):
            x = test_batch[t].unsqueeze(0)  # [1, 1, 16]
            _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        # prediction range : start prediction
        pred_steps = total_length - self.params.pred_start
        samples_high = torch.zeros(1, batch_size, pred_steps, device=device, requires_grad=False)  # [1, 1, 15616]
        samples_low = torch.zeros(1, batch_size, pred_steps, device=device, requires_grad=False)  # [1, 1, 15616]
        samples = torch.zeros(sample_times, batch_size, pred_steps, device=device, requires_grad=False)  # [99, 1, 15616]
        for j in tqdm(range(sample_times + 2)):
            for t in range(pred_steps):
                x = test_batch[self.params.pred_start + t].unsqueeze(0)  # [1, 1, 16]

                _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 1, 40], [2, 1, 40]
                # use h from all three layers to calculate mu and sigma
                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [1, 80]

                # Plan C:
                pre_beta_0 = self.pre_beta_0(hidden_permute)  # [1, 1]
                beta_0 = self.beta_0(pre_beta_0)  # [1, 1]
                pre_gamma = self.pre_gamma(hidden_permute)  # [1, 20]
                gamma = self.gamma(pre_gamma)  # [1, 20]

                # pred_cdf is a uniform distribution
                if j == 0:  # high
                    pred_cdf = torch.Tensor([cdf_high]).to(device)
                elif j == 1:  # low
                    pred_cdf = torch.Tensor([cdf_low]).to(device)
                else:
                    if sample:  # random
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0], device=device),
                            torch.tensor([1.0], device=device))
                        pred_cdf = uniform.sample([batch_size])  # [1, 1]
                    else:
                        pred_cdf = torch.Tensor([0.5]).to(device)

                sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [1, 20]
                beta = pad(gamma, (1, 0))[:, :-1]
                beta[:, 0] = beta_0[:, 0]
                beta = (gamma - beta) / (2 * sigma)
                beta = beta - pad(beta, (1, 0))[:, :-1]
                beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [1, 20]

                ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [1, 20]
                indices = ksi < pred_cdf  # [16, 20]
                pred = (beta_0 * pred_cdf).sum(dim=1)  # [1,]
                pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [1, 20]

                if j == 0:
                    samples_high[0, :, t] = pred
                elif j == 1:
                    samples_low[0, :, t] = pred
                else:
                    samples[j - 2, :, t] = pred

                # predict value at t-1 is as a covars for t,t+1,...,t+lag
                for lag in range(self.params.lag):
                    if t < pred_steps - lag - 1:
                        test_batch[self.params.pred_start + t + 1, :, 0] = pred

        sample_mu = torch.mean(samples, dim=0)  # [1, 15616]

        # move to cpu and covert to numpy for plotting
        sample_mu = sample_mu.squeeze()  # [15632]
        labels_batch = labels_batch.squeeze()  # [15632]
        samples_high = samples_high.squeeze()  # [15632]
        samples_low = samples_low.squeeze()  # [15632]

        sample_mu = sample_mu.detach().cpu().numpy()  # [15632]
        labels_batch = labels_batch.detach().cpu().numpy()  # [15632]
        samples_high = samples_high.detach().cpu().numpy()  # [15632]
        samples_low = samples_low.detach().cpu().numpy()  # [15632]

        # sample_mu is predicted value
        # labels_batch is true value
        # samples_high is high-probability value
        # samples_low is low-probability value
        from matplotlib import pyplot as plt

        plt.plot(sample_mu, label='Predicted Value', color='red')
        plt.plot(labels_batch, label='True Value', color='blue')
        plt.fill_between(range(pred_steps), samples_high, samples_low, color='gray', alpha=0.5)
        plt.title('Prediction')
        plt.legend()
        plt.savefig('prediction.png')
        # plt.show()


def loss_fn(func_param, labels: Variable):  # {[256, 1], [256, 20]}, [256,]
    beta_0, gamma = func_param  # [256, 1], [256, 20]
    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1], requires_grad=False)  # [256, 1], [256, 20]

    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
    beta[:, 0] = beta_0[:, 0]
    beta = (gamma - beta) / (2 * sigma)
    beta = beta - pad(beta, (1, 0))[:, :-1]
    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)

    # calculate the maximum for each segment of the spline
    ksi = torch.cumsum(sigma, dim=1)
    df1 = ksi.expand(sigma.shape[1], sigma.shape[0], sigma.shape[1]).T.clone()
    df2 = ksi.T.unsqueeze(2)
    ksi = pad(ksi, (1, 0))[:, :-1]
    knots = df1 - ksi
    knots[knots < 0] = 0
    knots = (df2 * beta_0).sum(dim=2) + (knots.pow(2) * beta).sum(dim=2)
    knots = pad(knots.T, (1, 0))[:, :-1]  # F(ksi_1~K)=0~max

    diff = labels.view(-1, 1) - knots
    alpha_l = diff > 0
    alpha_A = torch.sum(alpha_l * beta, dim=1)
    alpha_B = beta_0[:, 0] - 2 * torch.sum(alpha_l * beta * ksi, dim=1)
    alpha_C = -labels + torch.sum(alpha_l * beta * ksi * ksi, dim=1)

    # since A may be zero, roots can be from different methods.
    not_zero = (alpha_A != 0)
    alpha = torch.zeros_like(alpha_A)
    # since there may be numerical calculation error,#0
    idx = (alpha_B ** 2 - 4 * alpha_A * alpha_C) < 0  # 0
    diff = diff.abs()
    index = diff == (diff.min(dim=1)[0].view(-1, 1))
    index[~idx, :] = False
    # index=diff.abs()<1e-4#0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0#0
    alpha[idx] = ksi[index]  # 0
    alpha[~not_zero] = -alpha_C[~not_zero] / alpha_B[~not_zero]
    not_zero = ~(~not_zero | idx)  # 0
    delta = alpha_B[not_zero].pow(2) - 4 * alpha_A[not_zero] * alpha_C[not_zero]
    alpha[not_zero] = (-alpha_B[not_zero] + torch.sqrt(delta)) / (2 * alpha_A[not_zero])

    # formula for CRPS is here!
    crps_1 = labels * (2 * alpha - 1)
    crps_2 = beta_0[:, 0] * (1 / 3 - alpha.pow(2))
    crps_3 = torch.sum(beta / 6 * (1 - ksi).pow(4), dim=1)
    crps_4 = torch.sum(alpha_l * 2 / 3 * beta * (alpha.unsqueeze(1) - ksi).pow(3), dim=1)
    crps = crps_1 + crps_2 + crps_3 - crps_4

    crps = torch.mean(crps)
    return crps
