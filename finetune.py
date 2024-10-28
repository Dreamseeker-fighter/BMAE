#!/usr/bin/env python3
# encoding: utf-8

'''
@author: zengjunjie
@contact: zengjunjie@thinkenergy.tech
@software: Pycharm
@application:
@file: finetune.py
@time: 29/12/22 14:48
@desc:
'''

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import datetime

from loguru import logger
# from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import math
import sys
import time
from typing import Iterable, Optional
import pandas as pd
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from timm.data import Mixup
from timm.utils import accuracy
from pymongo import MongoClient

# client = MongoClient(host='10.30.10.10', port=8888)
client = MongoClient(host='10.30.10.11', port=27017, username='admin', password='admin123', authSource='admin')
db = client['RUL']
inference_collection = db['mae_inference_result']
collection = db['mae_result']

import util.misc as misc
import util.lr_sched as lr_sched

###RUL 的最大最小值，对数据进行归一化
# min_score= 1
# max_score= 800
# mean= 459.96733671903036
# std= 375.4825391841353
## 单一数据集调试
# mean= 405.03
# std=266.12


# ##五个数据集调试
# std=269.6887571706117
# mean= 420.3365616804715

std=484.1995139902539
mean=786.6058135072909

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        logger.info('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True).float()
        # vin = targets['vin']
        # print('label',targets['label'])
        RUL_label = (targets['label']-mean)/std
        RUL_label=RUL_label.to(device)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs.squeeze(-1), RUL_label.float())
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, label):
    criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_out = []
    all_target = []
    all_vins = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]['label']
        charge_file = batch[-1]['file']
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)
        # print(target)
        RUL_label =(target-mean)/std
        # RUL_label = (target-min_score)/(max_score-min_score)
        RUL_label=RUL_label.to(device)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.squeeze(-1), RUL_label.float())
        output_invert_norm=output*std+mean
        # output_invert_norm=output*(max_score-min_score)+min_score
        all_out.extend(list(output_invert_norm.squeeze(-1).cpu().detach().numpy()))
        # all_target.extend(list(target.cpu().numpy()))
        all_vins.extend(charge_file)
        metric_logger.update(loss=loss.item())

    df = pd.DataFrame(np.array([all_vins, all_out]).T, columns=['celltype', 'out'])
    df['out'] = df['out'].map(float)
    
    df_res = df.groupby("celltype")['out'].mean().reset_index()
    # logger.error(df_res)

    df_merge = pd.merge(df_res, label, left_on='celltype', right_on='file',how="inner")
    # df['label'] = df_merge['RUL'].astype(float)
    ##计算误差
    y_true=df_merge['RUL'].astype(float)
    y_pred=df_merge['out']
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(df_merge['RUL'].astype(float), df_merge['out'])
    nd = len(df_merge['RUL'])
    rmseca=np.sqrt(np.mean((df_merge['RUL'].astype(float) -df_merge['out'])**2))
    mseca=np.mean(np.square( df_merge['out']-df_merge['RUL'].astype(float)) )
    maeca = sum(abs(df_merge['RUL'].astype(float)[i] -df_merge['out'][i])  for i in range(nd)) / nd
    mapeca = sum(abs(df_merge['out'][i] -df_merge['RUL'].astype(float)[i] )/ df_merge['RUL'].astype(float)[i] for i in range(nd) if df_merge['RUL'].astype(float)[i] != 0) / nd * 100
    # print(f"true_RUL_{df_merge['RUL'].astype(float)}")
    # print(f"predict_RUL_{df_merge['out'].astype(float)}")
    # print(f"maxerror_{max(abs(df_merge['RUL'].astype(float)-df_merge['out'].astype(float)))}")
    # print(f"mseca_{mseca}")
    # print(f"mapeca_{mapeca}")
    # plt.plot(df_merge['RUL'],label='original')
    # plt.plot(df_merge['out'],label='prediction')
    # plt.xlabel('charge segments')
    # plt.ylabel('RUL/cycles')
    # plt.legend()
    # plt.show()
    # plt.savefig('/home/chenjianguo/mae_soh_estimation/finetune/result/result.png')
    # plt.close()
    mape= mean_absolute_percentage_error(df_merge['RUL'].astype(float), df_merge['out'])
    # logger.info("rmse: {},mape:{}".format(rmseca,mapeca))
    logger.info("rmse: {},mape:{}".format(rmse,mapeca))
    metric_logger.synchronize_between_processes()

    # logger.info(df)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, (rmseca,mapeca,maeca,mseca), df, df_merge
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, (rmseca,mapeca,maeca,mseca), df, df_merge
