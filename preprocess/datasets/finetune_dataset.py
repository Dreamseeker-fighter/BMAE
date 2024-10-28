# -*- coding: utf-8 -*-
# @Time : 2022/06/01 10:44
# @Author : zengjunjie
# @Email : zengjunjie@thinkenergy.net.cn
# @File : base.py
# @Project : anomaly_detection
import copy
import datetime
import multiprocessing
import os
from collections import OrderedDict
import pandas as pd
from loguru import logger
from .pre_utils import sampling
from .normalizer import Normalizer, Normalizer2
import json



class FunetuneBatteryData:
    """
    finetune电池dataset
    :param
    """

    def __init__(self, battery_data_args):
        """
        finetuneDate 类
        :param battery_data_args: dict
        """

        self.df_lst = {}  # 文件的个数
        self.data_lst = []
        self.battery_dataset = []
        self.segment_label = battery_data_args['segment_label']
        self.negative_sample_expanded = battery_data_args['negative_sample_expanded']
        self.label = battery_data_args['label']
        self.jobs = battery_data_args["jobs"]
        self.ram = battery_data_args["ram"]
        self.data_path = battery_data_args["data_path"]
        self.norm_path = battery_data_args["norm_path"]
        self.interpolate = battery_data_args["interpolate"]
        self.single_data_len = battery_data_args["single_data_len"]
        # self.train_columns = ['current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp',
        #        ] + ['mean_volt', 'std_volt', 'std_temp', 'mean_temp']
        self.train_columns =['timestamp','volt','current','cap','cycle','temp','quantity']

        self.check_dataset_files()
        self.mutilprocess_read()
        self.check_normalize()

    def check_dataset_files(self):
        # 获取文件列表，检测是否有数据文件
        """
        数据
        :return:
        """
        cars = pd.read_csv(self.label,dtype=object )['car']
        segment_df = pd.read_csv(self.segment_label,dtype=object)
        segments =list(segment_df['car'])
        self.segment_label_map = dict(zip(list(segment_df['car']),list(segment_df['label'])))
        for file in sorted(os.listdir(self.data_path)):
            segment_car = os.path.splitext(file)[0].split('_')[2] + "_" +  os.path.splitext(file)[0].split('_')[3]
            if (file.endswith('.csv') or file.endswith('.feather')) and (segment_car in list(
                cars)) and (os.path.splitext(file)[0][6:] in segments):
                self.data_lst.append(os.path.join(self.data_path, file))

        # self.data_lst = self.data_lst[:10]
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
            logger.info("loading datasets")
            for single_file_tuple in [return_list_tuple[0] for return_list_tuple in results]:

                self.battery_dataset.extend(single_file_tuple)
            logger.info("data length %d" % len(self.battery_dataset))
        except RuntimeError as e:
            logger.info(e)

    def check_normalize(self):
        # 加载定义好的norm.json
        norm_json = json.load(open(os.path.join(self.norm_path, "all_norm.json"), 'r'))
        self.normalizer = Normalizer2(params=norm_json)

    def read_file(self, file, cell_is_list=False):
        """

        读取csv或者feather的代码
        :param file:
        :return:
        """
        df = None
        file = os.path.join(self.data_path, file)
        # logger.info(file)
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.feather'):
            df = pd.read_feather(file)
        # if cell_is_list:
        #     df = self.columns_split(df, 'single_temp_list', 'temp_')
        #     df = self.columns_split(df, 'single_volt_list', 'volt_')
        # vlot_list = [col for col in df.columns if 'volt_' in col]
        # temp_list = [col for col in df.columns if 'temp_' in col]
        # """增加几列特征"""
        # df['mean_volt'] = df[vlot_list].mean(axis=1)
        # df['std_volt'] = df[vlot_list].std(axis=1)
        # df['std_temp'] = df[temp_list].std(axis=1)
        # df['mean_temp'] = df[temp_list].mean(axis=1)

        # retain_columns = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'mileage',
        #        'timestamp'] + ['mean_volt', 'std_volt', 'std_temp', 'mean_temp', ]
        retain_columns = ['timestamp','volt','current','cap','cycle','temp','quantity']
        df = df[retain_columns]
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
        file, time0, metadata = self.battery_dataset[index]
        if self.ram:
            df = self.df_lst[file]
        else:
            df = self.read_file(file)
            if self.interpolate:
                df = sampling(df, self.interpolate)
        df = df[self.train_columns]
        df = df.iloc[time0: time0 + self.single_data_len]
        df = df.values
        # 直接输出归一化的数据
        metadata['time'] = time0
        df = self.normalizer.norm(df)
        return df,  metadata


def pool_map(args):
    """
    多进程读取数据
    file 要读取的文件名
    """
    # try:
    self, file = args[0], args[1].split('/')[-1]

    return_lst = []
    metadata = OrderedDict()
    name = os.path.splitext(file)[0].split('_')
    metadata['vin'] = name[2] + "_" + name[3]
    # print(metadata['vin'])
    # metadata['label'] = 0 if os.path.splitext(file)[0].split('_')[0] == '00' and  os.path.splitext(file)[0].split('_')[1] == '00' else 1
    metadata['label'] = 0 if self.segment_label_map[os.path.splitext(file)[0][6:]] == '0' else 1
    df = self.read_file(file)  # 读取数据
    if self.interpolate:  # 对df进行插值处理
        df = sampling(df, self.interpolate)
    if df.shape[0] >= self.single_data_len:
        sequence_num = df.values.shape[0] // self.single_data_len
        for index in range(sequence_num):
            """样本增强"""
            return_lst.append((file, index * self.single_data_len, copy.deepcopy(metadata)))
            if metadata['label'] == 1 and self.negative_sample_expanded > 0:
                for i in range(self.negative_sample_expanded):
                    return_lst.append((file, index * self.single_data_len, copy.deepcopy(metadata)))

    if self.ram:
        return [return_lst, (file, df)]
    else:
        return [return_lst, ()]
