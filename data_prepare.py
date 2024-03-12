# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:17:40 2020

@author: 18096
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np


def format_data(y, covars, usage='train', lag=2, window_size=44, data_name='Zone1'):
    # features of each sample are composed of lags and covars
    num_covars = covars.shape[1]
    ts_len = y.shape[0]
    samples = ts_len - window_size + 1
    # lag表示滞后时间，在预测t时刻的时候，若lag=3，则会将t-1,t-2,t-3时刻的数据作为输入，放入数据维度的位置
    X = np.zeros((samples, window_size, lag + num_covars),
                 dtype='float32')  # [samples, window_size(seq_len+pred_len), dimension(lag+num_covars)]
    label = np.zeros((samples, window_size), dtype='float32')  # [samples, window_size(seq_len+pred_len)]
    for i in range(samples):
        # window_start=i
        window_end = i + window_size

        # 放入滞后变量到对应维度
        for j in range(lag):
            # add different z_(t-i)
            X[i, (j + 1):, j] = y[i:window_end - j - 1]

        # 放入相关变量到对应维度
        X[i, :, lag:] = covars[i:window_end, :]

        # 放入标签
        label[i, :] = y[i:window_end]

    save_path = os.path.join('data', data_name)
    prefix = os.path.join(save_path, usage + '_')
    np.save(prefix + 'X_' + data_name, X)
    np.save(prefix + 'label_' + data_name, label)


def gen_covars(df, ids_covar, fit_len):
    covars = np.zeros((df.shape[0], len(ids_covar)))
    count = 0
    for i in ids_covar:
        if i == -1:
            covars[:, count] = df.index.hour
        elif i == 0:
            covars[:, count] = df.index.month
        else:
            mean = df.iloc[:fit_len, i].mean()
            std = df.iloc[:fit_len, i].std()
            covars[:, count] = (df.iloc[:, i] - mean) / std
        count += 1
    return covars


def prepare_data(zone, lag=1, ids_covars=[1, 2, 3, 4], window_size=192):
    data_path = os.path.join('data', zone, zone + '.csv')

    df = pd.read_csv(data_path, sep=',', index_col=0, parse_dates=True)

    # 填充缺失值
    # df.fillna(0, inplace=True)

    # 处理极端数值
    # df.loc[df.TARGETVAR >= 1, 'TARGETVAR'] = 0.9999
    # df.loc[df.TARGETVAR <= 0, 'TARGETVAR'] = 0.0001

    # train test split
    seg = ['2012-01-01 01:00:00', '2013-05-01 00:00:00',
           '2013-04-27 01:00:00', '2013-08-01 00:00:00',
           '2013-07-28 01:00:00', '2014-01-01 00:00:00']

    # 生成相关变量，按照均值方差进行归一化
    covars = gen_covars(df, ids_covars, df[seg[0]:seg[1]].shape[0])

    y_train = df[seg[0]:seg[1]]['TARGETVAR'].values
    format_data(y_train, covars[:y_train.size], usage='train',
                lag=lag, window_size=window_size, data_name=zone)

    y_vali = df[seg[2]:seg[3]]['TARGETVAR'].values
    format_data(y_vali, covars[y_train.size - window_size + 12:][:y_vali.size], usage='vali',
                lag=lag, window_size=window_size, data_name=zone)

    y_test = df[seg[4]:seg[-1]]['TARGETVAR'].values
    format_data(y_test, covars[-y_test.size:], usage='test',
                lag=lag, window_size=window_size, data_name=zone)


def output_fillna_data(zone):
    data_path = os.path.join('data', zone, zone + '.csv')
    df = pd.read_csv(data_path, sep=',', index_col=0, parse_dates=True)
    df.fillna(0, inplace=True)
    df.to_csv(data_path)


if __name__ == '__main__':
    for i in range(10):
        output_fillna_data('Zone' + str(i + 1))
    prepare_data('Zone1', lag=3, ids_covars=[1, 2, 3, 4], window_size=96 + 12)
