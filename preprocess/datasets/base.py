# -*- coding: utf-8 -*-
# @Time : 2022/06/01 10:44
# @Author : zengjunjie
# @Email : zengjunjie@thinkenergy.net.cn
# @File : base.py
# @Project : anomaly_detection
import math
import multiprocessing
import os
import random
import numpy as np
import pandas as pd
from loguru import logger
from .pre_utils import sampling
from .normalizer import Normalizer, Normalizer2
import json
import glob



class BaseBatteryData:
    """
           初始化
           :param data_path: 数据路径 str
           :param patch_len: patch长度 int
           :param tokens_len: 序列长度 int
           :param interpolate: 数据最小间隔 int
           :param jobs: 进程数 int
           :param ram: 是否开启内存 bool
    """

    def __init__(self, battery_data_args):
        self.df_lst = {}  # 文件的个数
        self.data_lst = []
        self.battery_dataset = []
        self.jobs = battery_data_args["jobs"]
        self.ram = battery_data_args["ram"]
        self.data_path = battery_data_args["data_path"]
        self.norm_path = battery_data_args["norm_path"]
        self.interpolate = battery_data_args["interpolate"]
        self.single_data_len = battery_data_args["single_data_len"]
        # self.train_columns = ['current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp',
        #        ] + ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        # self.train_columns = ['timestamp','volt','current','cap','cycle','temp','quantity']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        # self.train_columns = ['timestamp','volt','current_C','cap','cycle','temp','quantity']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        self.train_columns = ['timestamp','volt','current_C','cap','cycle','temp','quantity_C']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        self.check_dataset_files()
        self.mutilprocess_read()
        self.check_normalize()

    def check_dataset_files(self):
        # 获取文件列表，检测是否有数据文件

        if type(self.data_path) is str:
            data_path=[self.data_path]#可视化时使用
            # data_path=data_path.split['[',']']
            
        else:
            data_path=self.data_path
        # print(data_path)
        for path in data_path:
            # logger.info(path)
            # print(sorted(os.listdir(path)))
            self.data_lst.extend([os.path.join(path,file) for file in sorted(os.listdir(path)) if
                        file.endswith('.csv') or file.endswith('.feather')])
            # print('aa',self.data_lst)
            # logger.info(self.data_lst)

        print("长度：", len(self.data_lst))
        logger.info("Using parallel loader for %d files; this takes about 1min per 50,000 files." % len(self.data_lst))
        if len(self.data_lst) == 0:
            print('.csv or .feather file not found')
            exit()

    def mutilprocess_read(self):
        """
        多进程读取数据
        """

        try:
            # 多进程读 ：
            if self.jobs == 1:
                results = [pool_map([self, file]) for file in self.data_lst]
            else:
                pool = multiprocessing.Pool(self.jobs)
                results = pool.map(pool_map, [[self, file] for file in self.data_lst])  # 调用pool_map函数
                pool.close()
                pool.terminate()
                pool.join()
            if self.ram:
                for file, df in [return_list_tuple[1] for return_list_tuple in results]:
                    self.df_lst[file] = df
            # print(self.df_lst[file])
            logger.info("loading datasets")
            for single_file_tuple in [return_list_tuple[0] for return_list_tuple in results]:
                self.battery_dataset.extend(single_file_tuple)
            logger.info("data length %d" % len(self.battery_dataset))
        except RuntimeError as e:
            logger.info(e)

    def check_normalize(self):
        # 加载定义好的norm.json
        norm_json = json.load(open(os.path.join(self.norm_path, "all_norm alldataCQC.json"), 'r'))# all_norm alldataCQC
        # self.normalizer = Normalizer(dfs=None, params=norm_json)
        self.normalizer = Normalizer2(params=norm_json)

    def read_file(self, file, cell_is_list=False):
        """

        读取csv或者feather的代码
        :param file:
        :return:
        """
        df = None
        # file='/home/chenjianguo/batterydata/Dataset_1_NCA_battery/Dataset_1_50slide/test/CY45-05_1-#28_621.csv'
        # file_abs_path = os.path.join(self.data_path, file)
        if file.endswith('.csv'):
            # print(file)
            df = pd.read_csv(file)
        elif file.endswith('.feather'):
            df = pd.read_feather(file)
        # if cell_is_list:
        #     df = self.columns_split(df, 'single_temp_list', 'temp_')
        #     df = self.columns_split(df, 'single_volt_list', 'volt_')
        # vlot_list = [col for col in df.columns if 'volt_' in col]
        # temp_list = [col for col in df.columns if 'temp_' in col]
        # # print(temp_list)
        # """增加几列特征"""
        # df['mean_volt'] = df[vlot_list].mean(axis=1)
        # df['std_volt'] = df[vlot_list].std(axis=1)
        # df['std_temp'] = df[temp_list].std(axis=1)
        # df['mean_temp'] = df[temp_list].mean(axis=1)
        # print(df['mean_temp'])

        # retain_columns = ['timestamp','volt','current','cap','cycle','temp','quantity']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        # retain_columns = ['timestamp','volt','current_C','cap','cycle','temp','quantity']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        retain_columns = ['timestamp','volt','current_C','cap','cycle','temp','quantity_C']#+ ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        df = df[retain_columns]
        # print(df['volt'])
        # print(df)
        return df

    def columns_split(self, dataframe, column_list_name, column_name):
        """ 拆分特定单列数据为多列数据

        Args:
            dataframe：dataframe格式的数据集
            column_list_name：需要拆分的特定单列数据名称
            column_name：拆分之后多列数据名称的前缀，例如：volt，temp等

        Returns：
            new_dataframe：所需维度名称标准化后的dataframe数据集
        """
        new_columns = [column_name + str(i + 1) for i in range(len(dataframe[column_list_name][0].split(";")))]
        dataframe[new_columns]= dataframe[column_list_name].str.split(';', expand = True)
        dataframe[new_columns] = dataframe[new_columns].astype(float)
        dataframe.drop(columns=[column_list_name], inplace=True)  # 删除需要拆分的特定单列数据名称
        return dataframe

    def __len__(self):
        """
        获取数据集长度
        """
        return len(self.battery_dataset)

    def __getitem__(self, index):
        """
        起到迭代器的作用，通过索引读取数据
        Args:
            index: 0，1，2......
        Returns:
            返回数据：df
        """
        file, time0 = self.battery_dataset[index]
        if self.ram:
            df = self.df_lst[file]

        else:
            df = self.read_file(file)
            if self.interpolate:
                df = sampling(df, self.interpolate)
        df = df[self.train_columns]
        df = df.iloc[time0: time0 + self.single_data_len]     
        # print('bb',df)   
        df = df.values
        # 直接输出归一化的数据
        df = self.normalizer.norm(df)##问题所在
        # print('aa',df)
        return df


def pool_map(args):
    """
    多进程读取数据
    file 要读取的文件名
    """
    # try:
    self, file = args[0], args[1]

    return_lst = []

    df = self.read_file(file)  # 读取数据
    # print(len(df))
    if self.interpolate:  # 对df进行插值处理
        df = sampling(df, self.interpolate)
    if df.shape[0] >=self.single_data_len:
        sequence_num = df.values.shape[0] // self.single_data_len
        for index in range(sequence_num):
            return_lst.append((file, index * self.single_data_len))
    if self.ram:
        return [return_lst, (file, df)]
    else:
        return [return_lst, ()]
    # except Exception as e:
    #     logger.info(e)
