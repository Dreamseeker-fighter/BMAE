import argparse
import datetime
import json
import socket

import pandas as pd
from loguru import logger
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import timm
from util.early_stop import EarlyStopping

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from preprocess.datasets.finetune_dataset import FunetuneBatteryData
from preprocess.datasets.finetune_RUL_dataset import FinetuneRULDataset
from finetune import train_one_epoch, evaluate, inference_collection
from finetune import collection
from finetune import train_one_epoch, evaluate

import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    parser.add_argument('--patch_len', default=1 )
    parser.add_argument('--tokens_len', default=49)
    parser.add_argument('--interval', default=10)
    parser.add_argument('--interpolate', default=3)
    parser.add_argument('--ram', default=True, )

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')



    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--nb_classes', default=1, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--anomaly_days', default=3600, type=int,
                        help='异常数据选择多少天的数据打标签')

    parser.add_argument('--desc',
                        default='',
                        type=str,
                        help='训练集标签路径')

    # Model parameters
    parser.add_argument('--model', default='battery_mae_vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--norm_path', default='./norm/', type=str,
                        help='dataset path')

    # * Finetuning params
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')


    parser.add_argument('--finetune', default='base pretrained model.pth',
                        help='finetune from checkpoint')

    # Dataset parameters
    
    
    
    parser.add_argument('--data_path',
                        default='datapath',
                        type=str,
                        help='dataset path')

    parser.add_argument('--label_path',
                        default='labelpath',
                        type=str,
                        help='训练集标签路径')
    # ##NCA Dateset
    NAME='timetest'
    
   
    parser.add_argument('--data_name',
                        default='NE',
                        type=str,
                        help='数据集名称')


    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')##finetune_CQC
    
    parser.add_argument('--output_dir', default='/home/{}/mae_soh_estimation/fintuneCQCep50/{}/model/'.format(os.getlogin(), now_time+NAME),
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/home/{}/mae_soh_estimation/fintuneCQCep50/{}/tensorboard/'.format(os.getlogin(), now_time+NAME),
                        help='path where to tensorboard log')
    parser.add_argument('--result_dir', default='/home/{}/mae_soh_estimation/fintuneCQCep50/{}/results/'.format(os.getlogin(), now_time+NAME),
                        help='path where to results data')

    parser.add_argument("--negative_sample_expanded", default=0, type=int,
                        help='负样本增强倍数')

    parser.add_argument("--reinit_last_block", default=2, type=int, help='是否强制初始化最后几层'

                        )
    parser.add_argument("--train_test_split_ratio", default=0.6, type=float, help='划分比例'
                        )
    return parser



def evaluate_test(args, finetune_model_path,val_file, label):
    """
    训练完成之后，找到最好的模型，直接infenrence
    :return:
    """
    dataset_test_eval = FinetuneSohDataset(data_path=args.data_path,
                                            file_list=val_file,
                                            label_path=args.label_path,
                                            window_len=args.tokens_len*args.patch_len,
                                            interval=args.interval,
                                            jobs=args.num_workers,
                                            ram=args.ram,
                                            interpolate=args.interpolate,
                                            norm_path=args.norm_path,
                                            data_name=args.data_name)
    sampler_train_val = torch.utils.data.SequentialSampler(dataset_test_eval)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test_eval, sampler=sampler_train_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # if args.finetune and not args.eval:
    checkpoint = torch.load(finetune_model_path, map_location='cpu')

    logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    model.to(args.device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model = %s" % str(model_without_ddp))
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # if args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.info("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    test_label = pd.read_csv(label, dtype=object)
    # if args.eval:
    test_stats, auc, df, df_res_test = evaluate(data_loader_test, model, args.device, test_label)
    # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    # test_stats,test_mongo_res,test_df = evaluate(data_loader_test, model, device)
    return test_stats, auc, df, df_res_test


def inerence_record(args, train_res, validate_res, test_res,
                    train_patches_res, validate_patches_res, test_patches_res,
                    train_cars_res, validate_cars_res, test_cars_res,
                    n_parameters, epoch, finetune_model
                    ):
    """"
    推理结果统一记录
    """
    inference_collection.insert_one(

        {
            "desc": args.desc,
            "host": socket.gethostname(),
            "author": os.getlogin(),
            "train_mse": train_res[0],
            "valid_mse": validate_res[0],
            "test_mse": test_res[0],
            "train_mape": train_res[1],
            "valid_mape": validate_res[1],
            "test_mape": test_res[1],
            "insert_time": datetime.datetime.now(),
            'epoch': epoch,
            "test_cars": len(test_cars_res),
            "train_cars": len(train_cars_res),
            "validate_cars": len(validate_cars_res),
            "test_patches": len(test_patches_res),
            "train_patches": len(train_patches_res),
            "validate_patches": len(validate_patches_res),
            'n_parameters': n_parameters,
            "base_model": args.finetune,
            "finetune_model": finetune_model,
            **vars(args)
        }
    )
    if not os.path.exists(os.path.join(args.output_dir, "inference")):
        os.makedirs(os.path.join(args.output_dir, "inference"), exist_ok=True)

    # train_cars_res.to_csv(os.path.join(args.output_dir, "inference") + "/train_vin_res.csv", index=0)
    # test_cars_res.to_csv(os.path.join(args.output_dir, "inference") + "/test_vin_res.csv", index=0)
    # validate_cars_res.to_csv(os.path.join(args.output_dir, "inference") + "/validate_vin_res.csv", index=0)
    # train_patches_res.to_csv(os.path.join(args.output_dir, "inference") + "/train_patches_res.csv", index=0)
    # test_patches_res.to_csv(os.path.join(args.output_dir, "inference") + "/test_patches_res.csv", index=0)
    # validate_patches_res.to_csv(os.path.join(args.output_dir, "inference") + "/validate_patches_res.csv", index=0)


def split_train_test_vin(args, data_list):
    file_name_set = set()
    datali=[]
    # print(data_list)
    for file in data_list:
        file_list = file.split('_')
        file_name = file_list[0] + '_' + file_list[1]
        
        # file_name = file_list[0] + '_' + file_list[1]+ '_' + file_list[2]## for NCMNCA dataset
        # print(file_name)
        file_name_set.add(file_name)
        
    # print(file_name_set)
    random.seed(42)
    train_vin = random.sample(list(file_name_set), int(len(file_name_set) * args.train_test_split_ratio))
    print('train_vin',len(train_vin))
    remain_vin = list(set(file_name_set).difference(set(train_vin)))# 这行代码从原始的file_name_set集合中移除已经分配给训练集的VIN，得到剩余的VIN集合。
    test_vin = remain_vin[::2]#这行代码从剩余的VIN集合中每隔一个VIN选取一个作为测试集。这里使用切片操作[::2]来实现。
    val_vin = remain_vin[1::2]#这行代码从剩余的VIN集合中每隔一个VIN选取一个作为验证集，与测试集不同的是，它从第二个VIN开始选取。这里同样使用切片操作[1::2]来实现。
    print('test_vin',len(test_vin))
    print('val_vin',len(val_vin))
    
    ## for NCMNCA data to finetune
    # b=np.array([data_list[n].split('_')[-1][:-4] for n in range(len(data_list))])
    # c=[data_list[n].split('_')[0]+'_'+data_list[n].split('_')[1] for n in range(len(data_list))]
    # list1 = [int(x) for x in b]
    # list1.sort()
    # dfn=[c[n]+'_'+str(list1[n])+'.csv' for n in range(len(c))]
    # print(dfn[0:50])
    # train_vin=dfn[0:int(len(dfn)*0.7)]
    # val_vin=dfn[int(len(dfn)*0.7):int(len(dfn)*0.85)]
    # test_vin=dfn[int(len(dfn)*0.85):int(len(dfn))]
    
    train_file=[]
    test_file = []
    val_file = []
    for file in data_list:
        file_list = file.split('_')
        file_name = file_list[0] + '_' + file_list[1]
        # file_name = file_list[0] + '_' + file_list[1]+ '_' + file_list[2]## for NCMNCA dataset
        if file_name in train_vin:
            train_file.append(file)
        elif file_name in test_vin:
            test_file.append(file)
        elif file_name in val_vin:
            val_file.append(file)
    # print(train_file)
    # print(val_file)
    # print(test_file)
    return train_file,val_file, test_file
    # return train_file[::15],val_file[::10], test_file[::10]


def get_data_list(args):
    """
    筛选包含label的充电段数据
    :param label_path:
    :param data_path: 所有充电段的存储路径
    """
    label_data = pd.read_csv(args.label_path, engine="python")
    # label_data["label_file"] = label_data["label_file"].apply(lambda file: str(file).replace("svolt1", "00"))
    car_lsit = set(os.listdir(args.data_path)) & set(label_data["file"].values)
    return list(car_lsit)


def main(args,files):
    misc.init_distributed_mode(args)#像是启动分布式训练模式的初始化步骤。
    ##以下两行用于记录程序的启动信息和配置参数，
    logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_file,val_file,test_file=files[0],files[1],files[2]
    ##训练数据集
    dataset_train = FinetuneSohDataset(data_path=args.data_path,
                                       file_list=train_file,
                                       label_path=args.label_path,
                                       window_len=args.tokens_len*args.patch_len,
                                       interval=args.interval,
                                       jobs=args.num_workers,
                                       ram=args.ram,
                                       interpolate=args.interpolate,
                                       norm_path=args.norm_path,
                                       data_name=args.data_name)
    ##测试数据集
    dataset_val = FinetuneSohDataset(data_path=args.data_path,
                                     file_list=test_file,
                                     label_path=args.label_path,
                                     window_len=args.tokens_len*args.patch_len,
                                     interval=args.interval,
                                     jobs=args.num_workers,
                                     ram=args.ram,
                                     interpolate=args.interpolate,
                                     norm_path=args.norm_path,
                                     data_name=args.data_name)
    ##验证数据集
    dataset_train_eval = FinetuneSohDataset(data_path=args.data_path,
                                            file_list=val_file,
                                            label_path=args.label_path,
                                            window_len=args.tokens_len*args.patch_len,
                                            interval=args.interval,
                                            jobs=args.num_workers,
                                            ram=args.ram,
                                            interpolate=args.interpolate,
                                            norm_path=args.norm_path,
                                            data_name=args.data_name)

    logger.info(dataset_train)
    #
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_train_val = torch.utils.data.SequentialSampler(dataset_train_eval)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    ## 加载训练数据
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    ## 加载验证数据
    data_loader_train_eval = torch.utils.data.DataLoader(
        dataset_train_eval, sampler=sampler_train_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    ##加载测试数据
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # block_length = len(model.blocks)
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info(msg)


        for k, v in model.named_children():
            logger.error({k: v})

        block_length = len(model.blocks)
        logger.error(block_length)
        if args.reinit_last_block > 0:
            for i in range(block_length - args.reinit_last_block, block_length):
                for name, param in model.blocks[i].named_parameters():
                    if 'attn' in name or 'mlp' in name:
                        logger.info(param)
                        nn.init.normal_(param, mean=0, std=0.02)

        for i in range(block_length - args.reinit_last_block, block_length):
            for name, param in model.blocks[i].named_parameters():
                if 'attn' in name or 'mlp' in name:
                    logger.info(param)
                    # print('param')
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    block_length = len(model.blocks)
    logger.error(block_length)
    if args.reinit_last_block > 0:
        for i in range(block_length - args.reinit_last_block, block_length):
            for name, param in model.blocks[i].named_parameters():
                if 'attn' in name or 'mlp' in name:
                    logger.info(param)
                    nn.init.normal_(param, mean=0, std=0.02)

    for i in range(block_length - args.reinit_last_block, block_length):
        for name, param in model.blocks[i].named_parameters():
            if 'attn' in name or 'mlp' in name:
                logger.info(param)
                # print('param')

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model = %s" % str(model_without_ddp))
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()

    logger.info("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_label = pd.read_csv(args.label_path, dtype=object)
    valid_label = pd.read_csv(args.label_path, dtype=object)
    best_epoch = train(data_loader_train, data_loader_val, data_loader_train_eval,
                       model, criterion, optimizer, device, loss_scaler, mixup_fn, log_writer,
                       model_without_ddp, n_parameters,
                       train_label, valid_label, args
                       )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    return best_epoch, n_parameters


def train(data_loader_train, data_loader_val, data_loader_train_eval,
          model, criterion, optimizer, device, loss_scaler, mixup_fn,
          log_writer, model_without_ddp, n_parameters, train_label, valid_label, args):
    early_stopping = EarlyStopping(patience=5)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        # best_rmse = float('inf')
        # best_epoch = -1
        # best_model = None
        
                
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        train_stats, train_mongo_res, train_df, df_res_train = evaluate(data_loader_train, model, device,
                                                                        train_label)
        valid_stats, valid_mongo_res, valid_df, df_res_valid = evaluate(data_loader_val, model, device, valid_label)
        # if valid_mongo_res[1] < best_rmse:
        #     # print(valid_mongo_res[1])
        #     best_rmse = valid_mongo_res[1] 
        #     best_epoch = epoch + 1
        #     best_metrics = valid_mongo_res
        #     # best_test_loss = test_loss / len(data_loader_val)
        #     best_model = model.state_dict()
        # if args.output_dir:
        #     misc.save_model(
        #         args=args, model=best_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=best_epoch)
        record(train_mongo_res, valid_mongo_res, epoch, n_parameters, train_df, valid_df, train_stats, train_stats,
               log_writer, df_res_train, df_res_valid)
        if early_stopping(valid_mongo_res[0], epoch):
            return epoch - 5
            # best_epoch==epoch - 5
            # return best_epoch
    return args.epochs - 6#args.epochs args.epochs - 6


def record(train_mongo_res, test_mongo_res, epoch,
           n_parameters, train_df, test_df, train_stats,
           test_stats, log_writer, df_res_train, df_res_test):
    collection.insert_one(

        {
            "desc": args.desc,
            "host": socket.gethostname(),
            "author": os.getlogin(),
            "train_mse": train_mongo_res[0],
            "valid_mse": test_mongo_res[0],
            "train_mape": train_mongo_res[1],
            "valid_mape": test_mongo_res[1],
            "insert_time": datetime.datetime.now(),
            'epoch': epoch,
            'train_pathes': len(train_df),
            'validate_patches': len(test_df),
            'train_cars': len(df_res_train),
            'validate_cars': len(df_res_test),
            'n_parameters': n_parameters,
            "base_model": args.finetune,
            **vars(args)
        }
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)    
    # train_df.to_csv(args.output_dir + "/train_res_{}.csv".format(epoch), index=0)
    # test_df.to_csv(args.output_dir + "/test_res_{}.csv".format(epoch), index=0)
    # df_res_train.to_csv(args.output_dir + "/train_res_all_{}.csv".format(epoch), index=1)
    # df_res_test.to_csv(args.output_dir + "/test_res_all_{}.csv".format(epoch), index=1)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters,
                 "base_model": args.finetune
                 }

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    # print('ee',args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data_list = get_data_list(args)
    # print(data_list)
    files= split_train_test_vin(args, data_list) # train_file,val_file,test_file
    # print(files)
    best_epoch, n_parameters = main(args,files)
    print('best_epoch',best_epoch)
    print('n_parameters',n_parameters)
    # best_epoch, n_parameters = 2, 1234
    finetune_model = os.path.join(
        args.output_dir, f'checkpoint-best.pth')#f'checkpoint-{best_epoch}.pth'

    if not os.path.exists(finetune_model):
        logger.info("因为多进程原因，创建的output_dir不匹配")
        wrong_output_dir = args.output_dir #'/log/zengjunjie/mae/2023-02-28-13-53-56/model/'
        time = wrong_output_dir.split('/')[-3]
        logger.error(time)
        prefix_path = '/'.join(wrong_output_dir.split('/')[:-3])
        datetime_time = datetime.datetime.strptime(time, '%Y-%m-%d-%H-%M-%S')
        for second in range(-3, 4):  # 正负3秒，一般差个1秒
            actual_time = datetime_time + datetime.timedelta(seconds=second)
            logger.info(actual_time)
            actual_time_str = actual_time.strftime('%Y-%m-%d-%H-%M-%S')
            actual_path = os.path.join(prefix_path, actual_time_str)

            logger.info(os.path.join(actual_path, 'model', f"checkpoint-{best_epoch}.pth"))
            if os.path.exists(os.path.join(actual_path, 'model', f"checkpoint-{best_epoch}.pth")):
                logger.error("find actual output path")
                finetune_model = os.path.join(actual_path, 'model', f"checkpoint-{best_epoch}.pth")##检查一下是保存最好的，还是保存所有数目{best_epoch}
                logger.info(finetune_model)
                break

    train_stats, train_res, train_patches_res, train_cars_res = evaluate_test(args, finetune_model,files[0], args.label_path)
    print(f"rmseca_train_{train_res[0]}")
    print(f"mapeca_train_{train_res[1]}")
    print('train_stats',train_stats)
    print('train_cars_res',train_cars_res)
    print('train_res',train_res)
    print('train_patches_res',train_patches_res)
    
    # plt.plot(range(len(train_cars_res)),train_cars_res['RUL'], marker='s', lw=1.0, ls='', c='red',label='origin')#
    # plt.plot(range(len(train_cars_res)),train_cars_res['out'].astype(int), marker='.', lw=1.0, ls='', c='blue',label='prediction')#
    # plt.xlabel('charge segments')
    # plt.ylabel('RUL/cycles')
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(args.result_dir) + 'resultrain.png')
    # plt.close()
    dftrain1 = pd.DataFrame(np.array(train_stats).flatten())
    dftrain2 = pd.DataFrame(train_res, index=['RMSE','MAPE','MAE','MSE'])
    dftrain3 = pd.DataFrame(train_patches_res)
    dftrain4 = pd.DataFrame(train_cars_res)
    
    dftrain1.to_csv(os.path.join(args.result_dir) + "/train_stats.csv", index=0)
    dftrain2.to_csv(os.path.join(args.result_dir) + "/train_res.csv")
    dftrain3.to_csv(os.path.join(args.result_dir) + "/train_patches_res.csv", index=0)
    dftrain4.to_csv(os.path.join(args.result_dir) + "/train_cars_res.csv", index=0)
    
    validate_stats, validate_res, validate_patches_res, validate_cars_res = evaluate_test(args, finetune_model,files[1],
                                                                                       args.label_path)
    print(f"rmseca_validate_{validate_res[0]}")
    print(f"mapeca_validate_{validate_res[1]}")
    print(validate_cars_res)
    print('stop')
    # plt.plot(range(len(train_cars_res)),train_cars_res['RUL'], marker='s', lw=1.0, ls='', c='red',label='origin')#
    # plt.plot(range(len(train_cars_res)),train_cars_res['out'].astype(int), marker='.', lw=1.0, ls='', c='blue',label='prediction')#
    # plt.xlabel('charge segments')
    # plt.ylabel('RUL/cycles')
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(args.result_dir) + 'resulval.png')
    # plt.close()
    dfval1 = pd.DataFrame(np.array(validate_stats).flatten())
    dfval2 = pd.DataFrame(validate_res, index=['RMSE','MAPE','MAE','MSE'])
    dfval3 = pd.DataFrame(validate_patches_res)
    dfval4 = pd.DataFrame(validate_cars_res)
    dfval1.to_csv(os.path.join(args.result_dir) + "/validate_stats.csv", index=0)
    dfval2.to_csv(os.path.join(args.result_dir) + "/validate_res.csv")
    dfval3.to_csv(os.path.join(args.result_dir) + "/validate_patches_res.csv", index=0)
    dfval4.to_csv(os.path.join(args.result_dir) + "/validate_cars_res.csv", index=0)
    
    test_stats, test_res, test_patches_res, test_cars_res = evaluate_test(args, finetune_model,files[2], args.label_path)
    print(f"rmseca_test_{test_res[0]}")
    print(f"mapeca_test_{test_res[1]}")
    print(test_cars_res)
    # plt.plot(range(len(train_cars_res)),train_cars_res['RUL'], marker='s', lw=1.0, ls='', c='red',label='origin')#
    # plt.plot(range(len(train_cars_res)),train_cars_res['out'].astype(int), marker='.', lw=1.0, ls='', c='blue',label='prediction')#
    # plt.xlabel('charge segments')
    # plt.ylabel('RUL/cycles')
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(args.result_dir) + 'resultest.png')
    # plt.close()
    dftest1 = pd.DataFrame(np.array(test_stats).flatten())
    dftest2 = pd.DataFrame(test_res, index=['RMSE','MAPE','MAE','MSE'])
    dftest3 = pd.DataFrame(test_patches_res)
    dftest4 = pd.DataFrame(test_cars_res)
    dftest1.to_csv(os.path.join(args.result_dir) + "/test_stats.csv", index=0)
    dftest2.to_csv(os.path.join(args.result_dir) + "/test_res.csv")
    dftest3.to_csv(os.path.join(args.result_dir) + "/test_patches_res.csv", index=0)
    dftest4.to_csv(os.path.join(args.result_dir) + "/test_cars_res.csv", index=0)
    inerence_record(
        args, train_res, validate_res, test_res,
        train_patches_res, validate_patches_res, test_patches_res,
        train_cars_res, validate_cars_res, test_cars_res,
        n_parameters, best_epoch, finetune_model
    )
