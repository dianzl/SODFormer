import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from engine import evaluate, train_one_epoch, concatenate_dataset
from models import build_spatial, build_temporal, build_fusion, build_cri_pro


def get_args_parser():
    parser = argparse.ArgumentParser('SODFormer Detector', add_help=False)

    # Data Loading
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # Setup
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # Variants of SODFormer
    parser.add_argument('--no_temporal', default=False, action='store_true',
                        help='If true, single frame detection will be implemented')
    parser.add_argument('--no_frame', default=False, action='store_true',
                        help='If true, only event stream will be used')
    parser.add_argument('--no_event', default=False, action='store_true',
                        help='If true, only frame stream will be used')
    parser.add_argument('--event_repre', default='image', type=str, choices=['image', 'voxel', 'gray'],
                        help='Event representation of event stream, disabled when no_event is set True')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the Spatial Transformer Encoder")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the Spatial Transformer Decoder")
    parser.add_argument('--tem_enc_layers', default=6, type=int,
                        help="Number of encoding layers in the Temporal Transformer Encoder")
    parser.add_argument('--tem_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the Temporal Transformer Decoder")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int, help='Number of sampling points in decoders')
    parser.add_argument('--enc_n_points', default=4, type=int, help='Number of sampling points in encoders')
    parser.add_argument('--n_frames', default=8, type=int, help='Temporal aggregation size')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of categories + 1')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # Dataset paths
    parser.add_argument('--frame_path', default='./data/aps_frames', type=str)
    parser.add_argument('--anno_path', default='./data/annotations', type=str)
    parser.add_argument('--event_path', default='./data/events_npys', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--categories_path', default='./datasets/neuromorphic_object_classes.txt')
    parser.add_argument('--spatial_aps_model', default='')
    parser.add_argument('--spatial_dvs_model', default='')
    parser.add_argument('--temporal_aps_model', default='')
    parser.add_argument('--temporal_dvs_model', default='')

    # Output paths
    parser.add_argument('--output_dir', default='./results/models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_method', default='SODFormer')
    parser.add_argument('--eval_file', default='SODFormer_eval.pth')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.no_frame:
        input_type = 'event_only'
    elif args.no_event:
        input_type = 'frame_only'
    else:
        input_type = 'multi-modal'

    criterion, postprocessors = build_cri_pro(args)
    if args.no_frame or args.no_event:
        if args.no_temporal:
            model = build_spatial(args)
        else:
            model = build_temporal(args, input_type)
    else:
        model = build_fusion(args)
    model.cuda()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, 
                                    momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            args.start_epoch = checkpoint['epoch'] + 1

    if args.no_frame or args.no_event:
        model_type = 'spatial' if args.no_temporal else 'temporal'
    else:
        model_type = 'fusion'

    collate_fn = utils.collate_fn
    if args.eval:
        dataset_test = concatenate_dataset(image_set='test', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_test = samplers.NodeDistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = samplers.DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=True,
                                      collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
        base_ds_test = utils.target_to_coco_format(data_loader_test, args.categories_path)
        coco_evaluator = evaluate(model, criterion, postprocessors, base_ds_test,
                                  data_loader_test, device, model_type, input_type)
        if args.output_dir:
            test_save_dir = output_dir / args.exp_method / 'test'
            Path(test_save_dir).mkdir(parents=True, exist_ok=True)
            test_save_path = test_save_dir / args.eval_file
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, test_save_path)
        return

    dataset_train = concatenate_dataset(image_set='train', args=args)
    dataset_val = concatenate_dataset(image_set='val', args=args)

    shuffle = True if args.no_temporal else False
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train, shuffle=shuffle)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=shuffle)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train, shuffle=shuffle)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=shuffle)
    else:
        if args.no_temporal:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.RandomSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.SequentialSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                        drop_last=True, collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                      drop_last=True, collate_fn=collate_fn, num_workers=args.num_workers,
                      pin_memory=True)
    base_ds_val = utils.target_to_coco_format(data_loader_val, args.categories_path)
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, model_type, input_type)
        lr_scheduler.step()

        evaluate(model, criterion, postprocessors, base_ds_val, data_loader_val, device, model_type, input_type)

        if args.output_dir:
            checkpoint_dir = output_dir / args.exp_method / 'train'
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'SODFormer_{epoch:02}.pth'
            if epoch == args.epochs - 1 or epoch % 5 == 0:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / args.exp_method / 'train' / "SODFormer.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SODFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
