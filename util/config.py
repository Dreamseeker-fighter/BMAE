#!/usr/bin/env python3
# encoding: utf-8

'''
@author: zengjunjie
@contact: zengjunjie@thinkenergy.tech
@software: Pycharm
@application:
@file: config.py
@time: 20/05/22 14:22
@desc:
'''
import sys
sys.path.insert(0, '../..')
sys.path = list(set(sys.path))
import os
import time
import yaml
from anomaly_detection.logger_config import logger
from anomaly_detection.model import projects
from anomaly_detection.utils import DotDict, config_valid, mkdir, copy_code


class Config:
    """
    配置类，读取配置文件，分门别类的设置各种参数，拆解self.args
    """

    def __init__(self,args):
        config_valid(args)
        self.args=args
        self.project_args = args.project_args
        #模型配置相关的参数
        self.path_config()
        self.model_params = args.model_params
        self.optuna_args = None
        self.project_args['study_name'] = None
        # 是否需要optuna
        if args.project_args.optuna:
            self.optuna_args = args.optuna_args
            self.project_args['study_name'] = self.optuna_args.study_name

        # 数据预处理相关的配置参数
        self.datapreprocess = args.datapreprocess
        self.loss_params = args.loss_params
        self.evaluation_params = args.evaluation_params
        self.project = projects.Project(self.project_args, self.datapreprocess)
        self.save()


    def path_config(self):
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

        """判断是否是需要fine-tune,这里需要在project_args模块指定current_path"""

        # 如果使用fine_tune进行微调
        if self.project_args.fine_tune:
            self.project_args.update({'path': self.project_args.exist_model_path})
            self.project_args.current_path = self.project_args.exist_model_path.current_path
            # 获取训练好的模型的文件路径
            self.project_args.previous_model_path = os.path.join(self.project_args.current_path ,'model')
            # 更新当前结果保存的路径
            self.project_args.current_path = os.path.join(self.project_args.current_path, 'fine_tune',self.project_args.save_model_path,time_now)
            self.path_variables()
        # 如果直接进行evaluate预测，
        elif self.project_args.evaluation_mode or self.project_args.extract_mode:
            self.project_args.update({'path': self.project_args.exist_model_path})
            self.project_args.current_path = self.project_args.exist_model_path.current_path
            self.project_args.current_model_path = os.path.join(self.project_args.current_path, 'model')
            path_name = ["loss_picture_path", "current_model_path", "result_path", "tb_path", "log_path",
                         "code_path"]
            path_name_values = ["loss", "model", "result", "tb", "log", "code"]
            for name, path in zip(path_name, path_name_values):
                self.project_args[name] = os.path.join(self.project_args.current_path, path)
        # 如果进行训练
        else:
            self.project_args.save_model_path = os.path.join(self.project_args.save_path, os.getlogin(),
                                                             self.project_args.save_model_path)
            self.project_args.current_path = os.path.join(self.project_args.save_model_path, time_now)
            mkdir(self.project_args.current_path)
            self.path_variables()
        # 如果自主划分train/test， train/test路径都更改为all_data_path, label 更改为all_label
        if self.project_args.divide_dataset_freely:
            self.project_args['path'].update({'evaluation_path': self.project_args.path.all_label_path})




    def path_variables(self):
        """
        新建保存模型的各种目录，并且增加属性
        """
        path_name = ["loss_picture_path", "current_model_path", "result_path", "tb_path","log_path", "code_path"
                     ]
        path_name_values = ["loss", "model", "result", "tb", "log", "code"]

        for name, path in zip(path_name, path_name_values):
            mkdir(os.path.join(self.project_args.current_path, path))
            self.project_args[name] = os.path.join(self.project_args.current_path, path)
        logger.info("train path create success!")



    def save(self):
        if self.project.project_args.save_code:
            """拷贝代码"""
            dest_path = self.project.project_args.code_path
            copy_code(dest_path)

