# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch
from preprocess.datasets.base import BaseBatteryData

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:25'



def get_args_parser():
    date = 'test_datasets'

    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='battery_mae_vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=25, metavar='N',
                        help='epochs to warmup LR')


    # data_path and log_path
    # parser.add_argument('--data_path', default=[
    #     '/data/nfsdata/database/JEVE/batch3/volt96_temp48/4ep_s10_d60_t30/all_data/feather',
    #     # '/data/nfsdata/database/jeve/batch1/car_type/volt96_temp48/regime_extract/charge_level/1ep_s10_d50_t40/new_dield_name_all_data/csv',
    #                                             '/data/nfsdata/database/biyadi/batch1/car_type/volt112_temp34/regime_extract/charge_level/1ep_s10_d50_t60/new_dield_name_all_data/feather',
    #                                             '/data/nfsdata/database/CALB/batch1/volt90_temp32/1ep_s10_d50_t30/all_data/feather',
    #                                             '/data/nfsdata/database/FENGCHAO/batch1/volt88_temp32/1ep_s10_d50_t60/all_data/feather',
    #                                             '/data/nfsdata/database/YEMA/batch1/volt90_temp20/1ep_s10_d50_t30/all_data/feather',
    #                                             '/data/nfsdata/database/CALB/batch1/volt92_temp32/1ep_s10_d50_t30/all_data/feather',
    #                                             '/data/nfsdata/database/CALB/batch2/volt90_temp32/1ep_s10_d50_t30/all_data/feather',
    #                                             '/data/nfsdata/database/wlhg_e50/batch3/all_data/csv'
    # ], type=list,
    parser.add_argument('--data_path', default=[

                                                '/home/chenjianguo/batterydata/propressdatsets/103-Tshall/train',
                                                '/home/chenjianguo/batterydata/propressdatsets/104-Severson_NEall/train',
                                                '/home/chenjianguo/batterydata/propressdatsets/124-TJ_Dataset_1_NCA_battery/train',
                                                '/home/chenjianguo/batterydata/propressdatsets/125-TJ_Dataset_2_NCM_battery/train',
                                                '/home/chenjianguo/batterydata/propressdatsets/126-TJ_Dataset_3_NCM_NCA_battery/train',
    ], type=list,
                        help='dataset path')
    parser.add_argument('--norm_path', default='/home/chenjianguo/mae_soh_estimation/norm/', type=str,
                        help='dataset path')
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument('--output_dir', default='/home/{}/mae_soh_estimation/pretrain/{}/model/'.format(os.getlogin(), now_time),
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/home/{}/mae_soh_estimation/pretrain/{}/tensorboard/'.format(os.getlogin(), now_time),
                        help='path where to tensorboard log')

    # battery data parameters
    # Dataset parameters
    parser.add_argument('--ram', default=True,)
    parser.add_argument('--patch_len', default=1)
    parser.add_argument('--tokens_len', default=49)#
    parser.add_argument('--interpolate', default=3)


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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

    return parser

# @profile
def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)


    misc.init_distributed_mode(args)

    logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))



    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # battery dataset args
    battery_data_args = {"data_path": args.data_path,
                         "norm_path": args.norm_path,
                         "patch_len": args.patch_len,
                         "ram": args.ram,
                         "single_data_len": args.patch_len * args.tokens_len,
                         "interpolate": args.interpolate,
                         "jobs": args.num_workers}
    dataset_train = BaseBatteryData(battery_data_args)
    logger.info(dataset_train)


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        logger.info("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    logger.info("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    logger.info(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # 每5 ecpoch 存一下
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
