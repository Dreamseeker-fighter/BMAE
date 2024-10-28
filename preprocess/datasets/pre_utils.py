#!/usr/bin/env python3
# encoding: utf-8

'''
@author: zengjunjie
@contact: zengjunjie@thinkenergy.tech
@software: Pycharm
@application:
@file: pre_utils.py
@time: 20/05/22 10:19
@desc:
'''


import numpy as np
import pandas as pd
from scipy import interpolate as ip



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


