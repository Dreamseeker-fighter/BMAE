#!/usr/bin/env python3
# encoding: utf-8



from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
from scipy import interpolate as ip
from torch.utils.data import Sampler


def sampling(df, interval=20, columns_name='timestamp'):
    """
    插值函数
    args:
        df: dataframe n * m
        interval: 插值间隔
        columns_name: 用哪列的间隔插值
    return: 
        new_df:插值后的df
    """
    df[columns_name] -= df[columns_name].min()
    target_time_idx = np.arange(df[columns_name].min() + 1,
                                df[columns_name].max() - 1,
                                interval)
    if len(target_time_idx) <= 10 or df.isnull().values.any():
        new_df = pd.DataFrame(df[:1])
        new_df.columns = list(df.columns)
        return new_df

    df = df.drop_duplicates(subset=[columns_name])  # 对df进行去重操作
    original_time = df[columns_name].values
    data_array = df.values
    f = ip.interp1d(original_time, data_array, axis=0)
    interpolated = f(target_time_idx)
    new_df = pd.DataFrame(interpolated)
    new_df.columns = list(df.columns)
    return new_df


def fc_delta_volt(df):
    """
    dv函数
    args:
        df: dataframe n * m
    return: 
        df:修改后的df
    """
    volt_index = df.columns[df.columns.str.contains('BMS_SingleCellVolt')]
    df[volt_index] = df[volt_index] - df[volt_index].mean(axis=1)
    return df


def fc_dv_dt(df):
    """
    args:
        df: dataframe n * m
    return: 
        df:修改后的df
    """
    dv_dt = df["BMS_BattVolt"].diff()
    dv_dt.iloc[0] = dv_dt.iloc[1]
    df["BMS_dvolt_dt"] = dv_dt
    return df


def fc_delta_temp(df):
    """
    args:
        df: dataframe n * m
    return: 
        df:修改后的df
    """
    temp_index = df.columns[df.columns.str.contains('BMS_BattModuleTemp')]
    df[temp_index] = df[temp_index] - df[temp_index].mean(axis=1)
    return df


def fc_dtemp_dt(df):
    """
    args:
        df: dataframe n * m
    return: 
        df:修改后的df
    """
    df["BMS_dtemp_dt"] = df["BMS_RMC_ModuleTempMax"] - df["BMS_RMC_ModuleTempMin"]
    return df


def fc_cell_level(df):
    """
    args:
        df: dataframe n * m
    return: 
        df:修改后的df
    """
    temp_index = df.columns[df.columns.str.contains('BMS_BattModuleTemp')]
    return df[temp_index]


def pad_tensor(vec, pad):
    """
    args:
        vec:tensor to pad
        pad:the size to pad to
    return:
        a new tensor padded to 'pad'
    """
    return torch.cat([vec, torch.zeros((pad - len(vec), vec.shape[-1]), dtype=torch.float)], dim=0).data.numpy()


def collate(batch_data):
    """
    collate 用来确定dataloader生成batch的方式，这里用来对不同长度的序列的padding，并进行排序
    args:
        batch_data - list of (tensor, metadata)

    return:
        (padded_sent_seq, data_lengths), metadata

    """

    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    seq_lengths = [len(xi[0]) for xi in batch_data]
    max_len = max(seq_lengths)

    sent_seq = [torch.FloatTensor(v[0]) for v in batch_data]
    # logger.info('sent_seq:/n',sent_seq)
    metadata_list = [xi[1] for xi in batch_data]  # 处理metadata，将dict合并起来
    metadata = OrderedDict([('label', []), ('vin', []), ('charge_segment', []), ('mileage', []), ('timestamp', [])])
    for i in range(len(metadata_list)):
        for key, value in metadata_list[i].items():
            metadata[key].append(value)

    padded_sent_seq = torch.FloatTensor([pad_tensor(v, max_len) for v in sent_seq])
    metadata['seq_lengths'] = seq_lengths
    return padded_sent_seq, metadata


class PartSampler(Sampler):
    """
    Dataloader的参数，怎么选取索引
    """
    def __init__(self, lst, cellnum):
        self.cellnum = cellnum
        self.lst = [i * self.cellnum for i in lst]
        self.a = []

    def __iter__(self):
        self.a = []
        sum = 0
        for i in range(len(self.lst)):
            for index in random.sample(range(sum, sum + self.lst[i]), self.lst[i]):
                self.a.append(index)
            sum += self.lst[i]
        return iter(self.a)

    def __len__(self):
        return len(self.a)
