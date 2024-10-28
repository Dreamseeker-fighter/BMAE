# -*- coding: utf-8 -*-
# @Time : 2022/07/05 10:44
# @Author : huangshaobo
# @Email : huangshaobo@thinkenergy.net.cn
# @File : utils.py
# @Project : anomaly_detection
import json
import os
import shutil
import time
import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
from util.logger_config import logger
from functools import wraps
from pydantic import BaseModel
from pydantic.typing import Literal
from pymongo import MongoClient
from pathlib import Path

client = MongoClient(host='gpu002', port=27017)
db = client['anomaly_detection']
collection = db['experiment_result']
BASE_DIR = Path(__file__).resolve().parent.parent

class DotDict(dict):
    """
    字典包装类，让字典可以用dot语法直接获取属性值
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                value = DotDict(value)
        except KeyError:
            raise AttributeError
        return value




def mkdir(path):
    """
    创建目录
    :param path: 要创建的路径
    """
    if os.path.exists(path):
        logger.info('%s is exist' % path)
    else:
        os.makedirs(path)


class ProjectArgs(BaseModel):
    """:cvar
    项目的配置文件，
    """
    project: str  #项目名称
    fine_tune: bool   #是否是fine_tune
    optuna: bool      #是否需要optuna超参数寻优
    norm: str  # 正则化
    load_dbfile: bool  # 训练
    save_model_path: str  # 存储模型前缀
    save_code: bool       #是否保存代码，默认保存
    cell_level: bool      #是单体级别还是整块电池的异常检测
    device: Literal['cpu', 'cuda']          # cuda或者cpu
    task: str              # task类型，可以限定一下
    partition: str        # 是否分区
    cell_level: bool   # 是否是单体级别
    divide_dataset_freely: bool
    '''divide_dataset_freely参数， 自主划分train/test数据集, 需要和以下参数配合使用：
    1. train_normal_car_num, 指定训练正常车的数量，其他数据作为测试集，不能选"all"，否则全部正常车会用来训练，导致测试集没有正常车
    2. evaluation_test_only设置为true, 只需要评估测试集就可以
    3. 需要填写all_data_path和all_label_path'''



class OptunaParams(BaseModel):
    """:cvar
    optuna的超参数寻优
    """
    learning_rate: list     #
    epochs: list
    cosine_factor: list
    # noise_scale: list
    anneal0: list
    nll_weight: list
    latent_label_weight: list
    percentile: list
    contamination: list
    use_flag:list  # "rec_error","copod_score", "IF_score", "svm_score"

    """
    use_flag 使用说明：
    1. use_flag 根据抽取的特征进行得分的计算，有多重可以选择的方法
    2. 在optuna中使用的时候，"rec_error","copod_score", "IF_score"可以放在一起，svm_score
        只能单独使用
    3. 关于svm_score的介绍：
        3.1  svm_score在训练集上训练一个svm model， 在测试集进行测试
        3.2  需要划分训练测试，训练测试都有故障车，divide_dataset_freely: false
        3.3  optuna 返回值为 测试集AUC
        3.4  svm模型保存在model 路径， grsearch.model
    """



class OptunaArgs(BaseModel):
    """:cvar
    optuna配置校验类
    """
    study_name: str  # 每次训练需要给一个新的名字
    n_trials: int  # optuna巡游次数
    params: OptunaParams



class DataPreprocess(BaseModel):

    """:cvar
    数据预处理校验类

    """
    window_len: int
    interval: int
    interpolate: int
    ram: bool
    variable_length: bool
    min_length: int
    jobs: int
    train_normal_car_num: int



class ModelParams(BaseModel):
    """:cvar
    模型参数校验类
    """
    hidden_size: int
    latent_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    cosine_factor: float
    model_type: str   # 模型类型
    rnn_type: str
    num_layers: int
    kernel_size: int
    bidirectional: bool
    nhead: int
    dim_feedforward: int
    noise_scale: float



class LossParams(BaseModel):
    """:cvar
    权重校验类
    """
    anneal_function: str # anneal更新函数
    anneal0: float   #初始kl权重系数
    k: float
    x0: int
    nll_weight: float  #nll loss权重
    latent_label_weight: float


class EvaluateParams(BaseModel):
    """:cvar
    评估检出参数校验
    """
    num_granularity_all: int
    num_granularity_car: int
    use_flag: str
    percentile: float
    granularity_all: int
    granularity_car: int
    contamination: float


def config_valid(args):
    """
    逐个验证config每个字段
    :param config: 读取到的config（包括config.json以及parser）
    :return: True or False
    """
    project_args = ProjectArgs(**args.project_args)
    optuna_args = OptunaArgs(**args.optuna_args)
    datapreprocess = DataPreprocess(**args.datapreprocess)
    model_params= ModelParams(**args.model_params)
    loss_params = LossParams(**args.loss_params)
    evaluate_params = EvaluateParams(**args.evaluation_params)
    logger.info(project_args.json)
    logger.info(optuna_args.json)
    logger.info(datapreprocess.json)
    logger.info(model_params.json)
    logger.info(loss_params.json)
    logger.info(evaluate_params.json)
    if datapreprocess.train_normal_car_num == -1 and project_args.divide_dataset_freely:
        logger.error("一个数据集不可设置train_normal_car_num 为-1 ！")
        raise ValueError("一个数据集不可设置train_normal_car_num 为-1 ！")
    if evaluate_params.use_flag == "svm_score" and project_args.divide_dataset_freely:
        logger.error("计算svm_score 训练集测试集均要有故障车")
        raise ValueError("计算svm_score 训练集测试集均要有故障车")



def to_var(x, device='cpu'):
    """
    如果有gpu将x放入cuda中
    :param x: data or model
    :param device cpu / gpu
    :return: 放入cuda后的x
    """
    if device == 'cuda':
        x = x.cuda()
    return x

def timer(func):
    """
    装饰器,打印函数运行时间
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        begin_time = time.time()
        result = func(*args, **kwargs)
        start_time = time.time()
        logger.info('func:{} args:[{}, {}] took: {} sec'.format(func.__name__, args, kwargs, start_time - begin_time))
        return result
    return wrap


def copy_code(dest_path):
    """
    存储代码模块,压缩当前版本代码并移动到code存储路径
    :param dest_path: save code path
    """
    if os.path.exists(dest_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(dest_path)
    shutil.copytree(BASE_DIR, dest_path)



def update_previous_model_params(model_path):
    """
    更新model_params.json文件的参数
    """
    model_params_path = os.path.join(model_path, "model_params.json")
    logger.info(model_params_path)
    with open(model_params_path, 'r') as load_f:
        prams_dict = json.load(load_f)
    logger.info(prams_dict)
    model_params = prams_dict['args']
    return model_params

