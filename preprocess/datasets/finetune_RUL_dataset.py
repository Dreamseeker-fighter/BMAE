#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: xionggengang,jianguochen
@file: finetune_soh_dataset.py
@time: 2023/3/13 17:33
"""

import os
import numpy as np
import pandas as pd
import multiprocessing
import random
from scipy import interpolate as ip
from collections import OrderedDict
from decimal import Decimal
from .normalizer import Normalizer, Normalizer2
import json
class FinetuneRULDataset:
    """
    电池滑窗数据类
    """

    def __init__(self, data_path, file_list, label_path, window_len, interval, jobs, ram,  interpolate,
                 data_name,norm_path, partition=False):
        """
        初始化
        :param data_path: 数据路径 str
        :param file_list: 文件列表 list
        :param label_path: 数据路径 str
        :param window_len: 窗长 int
        :param interval: 窗移 int
        :param jobs: 进程数 int
        :param ram: 是否开启内存 bool

        :param partition: 是否分段读取 bool

        """

        self.data_path = data_path
        self.label_path = label_path
        self.label_data=pd.read_csv(label_path,engine="python",index_col="file")
        # print('ee',self.label_data)
        self.battery_dataset = []
        # self.data_temp_lst = sorted(os.listdir(data_path))
        self.data_temp_lst = file_list
        if len(self.data_temp_lst) == 0:
            print('.csv or .feather file not found')
            exit()
        self.window_len = window_len
        self.interval = interval
        self.ram = ram
        self.data_name=data_name

        self.norm_path = norm_path
        self.interpolate = interpolate

        self.label_df = None
        self.partition = partition  ####分段load开关
        # self.column_filter = ['current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp'
        #        ] + ['mean_volt', 'std_volt', 'std_temp', 'mean_temp'] # ,'quantity','dqdv'
        # self.column_filter =['timestamp','volt','current','cap','cycle','temp','quantity']
        self.column_filter =['timestamp','volt','current_C','cap','cycle','temp','quantity_C']
        self.data_lst = [i for i in self.data_temp_lst]
        self.p_datapath = self.data_lst
        self.numlst = None
        self.check_normalize()
        # try:
        if jobs == 1:
            results = [self.pool_map([self, file]) for file in self.p_datapath ]
        else:
            pool = multiprocessing.Pool(jobs)

            results = pool.map(FinetuneSohDataset.pool_map, [[self, file] for file in self.p_datapath ])

            pool.close()
            pool.terminate()
            pool.join()

        for res in results:
            for lst in res:
                self.battery_dataset.extend(lst)
        print("data length %d" % len(self.battery_dataset))
        # except RuntimeError as e:
        #     print(e)
    def check_normalize(self):
        # 加载定义好的norm.json
        norm_json = json.load(open(os.path.join(self.norm_path, "all_norm alldataCQC.json"), 'r'))
        self.normalizer = Normalizer2(params=norm_json)
    def columns_merge(self,df, target_name_volt, target_name_temp):
        all_col_list = list(df.columns)
        sp_volt_list = []
        sp_temp_list = []
        for col in all_col_list:
            if col.startswith(f"volt") and col[-1].isdigit():
                df[col] = df[col].apply(lambda x: Decimal(str(x)).quantize(Decimal('0.000'))).astype('string')
                sp_volt_list.append(col)
            elif col.startswith(f"temp") and col[-1].isdigit():
                df[col] = df[col].apply(lambda x: Decimal(str(x)).quantize(Decimal('0.0'))).astype('string')
                sp_temp_list.append(col)
        df[target_name_volt] = df[sp_volt_list[0]].str.cat(
            [df[colu] for colu in sp_volt_list if colu != sp_volt_list[0]], sep=';')
        df[target_name_temp] = df[sp_temp_list[0]].str.cat(
            [df[colu] for colu in sp_temp_list if colu != sp_temp_list[0]],
            sep=';')
        df = df.drop(sp_volt_list + sp_temp_list, axis=1)
        return df
    def add_other_columns(self,df):
        if "single_volt_list" not in df.columns:
            df = self.columns_merge(df, "single_volt_list", "single_temp_list")
        df["mean_volt"]=df["single_volt_list"].apply(lambda x: round(np.mean([float(i) for i in x.split(";")]),2))
        df["mean_temp"] = df["single_temp_list"].apply(lambda x: round(np.mean([float(i) for i in x.split(";")]),2))
        df["std_volt"]=df["single_volt_list"].apply(lambda x: round(np.std([float(i) for i in x.split(";")]),2))
        df["std_temp"] = df["single_temp_list"].apply(lambda x: round(np.std([float(i) for i in x.split(";")]),2))

        # df.loc[0,"quantity"]=0

        return df[self.column_filter]
    def dqdv(self,df):
        a = [0]
        for i in range(len(df)-1):
            res = (df['quantity'][i+1] - df['quantity'][i])/(df['volt'][i+1] - df['volt'][i])
            a.append(res)
        return np.array(a)

    @staticmethod
    def pool_map(args):
        """
        多进程读取数据
        file 要读取的文件名
        """
        # try:
        self, file = args[0], args[1]
        return_lst = []
        df = pd.read_csv(os.path.join(self.data_path, file),engine='python')
        # print(df.columns)
        # print(file)
        # df=self.add_other_columns(df)
        # df=df.fillna(0)
        # # print(df.columns)
        # if self.interpolate:
        #     df = sampling(df, self.interpolate)


        name = os.path.splitext(file)[0].split('_')
        # print('label',name)
        metadata = OrderedDict()
        try:
            metadata['label'] = round(self.label_data.loc[file, "RUL"], 0)
        except:
            start_str=file[:5]
            file=file.replace(start_str,"svolt1_00")
            metadata['label'] = round(self.label_data.loc[file, "RUL"], 0)
        # print('ee',metadata['label'])
        metadata["file"]=file
        cell_df = pd.DataFrame(df, columns=self.column_filter)

        cell_df = cell_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        # print('ee',cell_df.columns)
        cell_df = self.normalizer.norm(cell_df)
        if cell_df.shape[0] >= self.window_len:
            window_num = int((cell_df.values.shape[0] - self.window_len) / self.interval) + 1
            index = 0
            while index < window_num:
                return_lst.append((np.array(
                    cell_df.iloc[index * self.interval:index * self.interval + self.window_len, :]),
                                   metadata))
                index += 1


        return [return_lst]

        # except Exception as e:
        #     print(e)
    def __len__(self):
        # 返回文件数据的数目
        if self.partition :
            return sum(self.numlst)
        else:
            return len(self.battery_dataset)
    def getnumlst(self):
        return self.numlst
    def __getitem__(self, idx):
        if self.partition:
            if idx>=self.amount[self.p_num+1] or (idx<self.amount[1] and self.p_num==(self.p-1)):
                if idx<self.amount[1] and self.p_num==(self.p-1):
                    self.p_num=0
                else:
                    self.p_num+=1
                # print ("loading next partition %s"%self.p_num)
                self.p_datapath = self.data_lst[self.p_num*self.p_len:(self.p_num+1)*self.p_len]
                results=[]
                self.battery_dataset=[]
                self.df_lst = {}
                for file in self.p_datapath:
                    results.append(FinetuneSohDataset.pool_map([self, file]))
                for lst in [i for i in results]:
                    self.battery_dataset.extend(lst)
            idx = idx-self.amount[self.p_num]
            if idx >= len(self.battery_dataset):
                #print(idx)
                idx = random.randint(0,len(self.battery_dataset))
###############################################################
        sig_data, label = self.battery_dataset[idx]
        return sig_data, label



class PreprocessNormalizer:
    """
    数据归一化类
    """

    def __init__(self, dataset, norm_name=None, normalizer_fn=None):
        """
        初始化
        :param dataset: SlidingWindowBattery
        :param norm_name: 用哪种归一化 如 ev 表示 EvNormalizer
        :param normalizer_fn: 归一化函数
        """
        self.dataset = dataset
        self.norm_name = norm_name
        self.normalizer_fn = normalizer_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        df, label = self.dataset[idx][0], self.dataset[idx][1]
        if self.normalizer_fn is not None:
            df = self.normalizer_fn(df, self.norm_name)
        return df, label

    def get_column(self):
        df = self.dataset[1][0]
        return list(df.columns)


def sampling(df, interval=20, columns_name='timestamp'):
    """
    插值函数
    :param df: dataframe n * m
    :param interval: 插值间隔
    :param columns_name: 用哪列的间隔插值
    :return: 插值后的df
    """

    df[columns_name] -= df[columns_name].min()
    target_time_idx = np.arange(df[columns_name].min() + 1,
                                df[columns_name].max() - 1,
                                interval)

    if len(target_time_idx) <= 10 or df.isnull().values.any():
        new_df = pd.DataFrame(df[:1])
        new_df.columns = list(df.columns)
        return new_df

    df = df.drop_duplicates(subset=[columns_name])
    original_time = df[columns_name].values
    data_array = df.values
    f = ip.interp1d(original_time, data_array, axis=0)
    interpolated = f(target_time_idx)
    new_df = pd.DataFrame(interpolated)
    new_df.columns = list(df.columns)

    return new_df

