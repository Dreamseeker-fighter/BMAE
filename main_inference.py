import argparse
import datetime
import json
import socket

import pandas as pd
from loguru import  logger
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
assert timm.__version__ == "0.3.2"  # version check
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from preprocess.datasets.finetune_dataset import FunetuneBatteryData
from finetune import evaluate, inference_collection


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    parser.add_argument('--patch_len', default=2, )
    parser.add_argument('--tokens_len', default=49)
    parser.add_argument('--interpolate', default=30)
    parser.add_argument('--ram', default=True, )

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')



    parser.add_argument('--input_size', default=49, type=int,
                        help='images input size')

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

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--seed', default=0, type=int)
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

    parser.add_argument('--anomaly_days',default=3600, type=int,
                        help= '异常数据选择多少天的数据打标签')

    parser.add_argument('--desc',
                        default='',
                        type=str,
                        help='训练集标签路径')

    # Model parameters
    parser.add_argument('--model', default='battery_mae_vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')


    parser.add_argument('--norm_path', default='/home/zengjunjie/mae/norm/', type=str,
                        help='dataset path')

    # * Finetuning params
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # parser.add_argument('--finetune', default='/data/log/zengjunjie/mae/model/2023-01-05-14-09-32/checkpoint-8.pth',
    #                     help='finetune from checkpoint')

    parser.add_argument('--finetune', default='/data/nfsdata005/checkpoint-99.pth',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path',
                        default='data/nfsdata/database/FENGCHAO/batch1/volt88_temp32/1ep_s10_d50_t60/all_data/feather',
                        type=str,
                        help='dataset path')



    parser.add_argument('--test_label', default='/data/nfsdata/database/FENGCHAO/batch1/volt88_temp32/1ep_s10_d50_t60/train_test/1sp/label/test_label.csv', type=str,
                        help='测试集标签路径')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument('--output_dir', default='/log/{}/mae/{}/model/'.format(os.getlogin(), now_time),
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/log/{}/mae/{}/tensorboard/'.format(os.getlogin(), now_time),
                        help='path where to tensorboard log')

    parser.add_argument("--negative_sample_expanded", default=0, type=int,
                        help='负样本增强倍数')
    parser.add_argument("--segment_label", default='/home/zengjunjie/mae4/norm/charge_segment_label.csv', type=str,
                        help='片段标签')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    logger.error(args)

    logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True



    battery_data_args_test = {"data_path": args.data_path,
                             'negative_sample_expanded': 0,
                             "label": args.test_label,
                             "norm_path": args.norm_path,
                             "patch_len": args.patch_len,
                             "ram": args.ram,
                             "single_data_len": args.patch_len * args.tokens_len,
                             "interpolate": args.interpolate,
                             "jobs": args.num_workers,
                             "anomaly_days": args.anomaly_days,
                              "segment_label": args.segment_label
                             }


    dataset_test = FunetuneBatteryData(battery_data_args_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')

            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias

        else:

           sampler_test = torch.utils.data.SequentialSampler(dataset_test)


    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
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

    # if args.finetune and not args.eval:
    checkpoint = torch.load(args.finetune, map_location='cpu')

    logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
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

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    logger.info("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    test_label = pd.read_csv(args.test_label,dtype=object)
    # if args.eval:
    test_stats,auc,df,df_res_test = evaluate(data_loader_test, model, device, test_label)
    # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    # test_stats,test_mongo_res,test_df = evaluate(data_loader_test, model, device)
    inference_collection.insert_one(

        {
            "desc": args.desc,
            "host": socket.gethostname(),
            "author": os.getlogin(),
            "test_auc": auc,
            "insert_time":datetime.datetime.now(),
            # 'epoch': epoch,
            "test_cars": len(df_res_test),
            'n_parameters': n_parameters,
            "base_model": args.finetune,
            **vars(args)
        }
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    df.to_csv(args.output_dir + "/test_res.csv", index=0)
    df_res_test.to_csv(args.output_dir + "/test_res_vin.csv", index=1)

    log_stats = {**{f'train_{k}': v for k, v in test_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 # 'epoch': epoch,
                 'n_parameters': n_parameters,
                 "base_model": args.finetune,
                 # "test_cars": len()
                 }

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
