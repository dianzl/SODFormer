import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from engine_asyn import dataset_single, evaluate
from models import build_spatial, build_temporal, build_fusion_asyn, build_cri_pro
import cv2
import random


def get_args_parser():
    parser = argparse.ArgumentParser('SODFormer Detector Visualization', add_help=False)
    
    # Data Loading
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--scene', default='normal', type=str, choices=['normal', 'low_light', 'motion_blur'])
    parser.add_argument('--datasetname', default=1, type=int)

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
    parser.add_argument('--event_repre', default='image', type=str, choices=['image', 'voxel'],
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
    parser.add_argument('--event_path', default='./data/asyn/events_npys', type=str)
    parser.add_argument('--back_path', default='./data/asyn/davis_images', type=str,
                        help='path to the background images')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--categories_path', default='./datasets/neuromorphic_object_classes.txt')
    parser.add_argument('--spatial_aps_model', default='')
    parser.add_argument('--spatial_dvs_model', default='')
    parser.add_argument('--temporal_aps_model', default='')
    parser.add_argument('--temporal_dvs_model', default='')

    # Output paths
    parser.add_argument('--vis_dir', default='results/vis',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_method', default='SODFormer')
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
        model = build_fusion_asyn(args)
    model.cuda()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

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

    if args.no_frame or args.no_event:
        model_type = 'spatial' if args.no_temporal else 'temporal'
    else:
        model_type = 'fusion'

    collate_fn = utils.collate_fn

    scene, dataset_test, dataset_name = dataset_single(image_set='test', args=args)
    print('Visualizing test video {} of scene {}'.format(dataset_name, scene))
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
    prediction = evaluate(model, criterion, postprocessors,
                          data_loader_test, device, model_type, input_type)

    back_path = os.path.join(args.back_path, 'test', scene, dataset_name)
    save_path = os.path.join(args.vis_dir, scene, dataset_name, args.exp_method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    colors = [[205, 197, 122],[197, 193, 205],[143, 143, 188]]
    tl = 2  # line/font thickness
    classes = ['car', 'person', 'two-wh']
    
    for i in range(len(prediction)):
        # print('Processing frame {}.png'.format(i))
        print('\r', end='')
        print('Processing Progress: {:.2%}'.format(i / len(prediction)), '▋' * (i * 50 // len(prediction)), end='')
        sys.stdout.flush()
        frame = cv2.imread(os.path.join(back_path, str(i)+'.png'))
        _, pred = prediction[i]
        scores = pred['scores']
        labels = pred['labels']
        boxes = pred['boxes']

        obj_idx = labels < 3

        obj_scores = scores[obj_idx]
        obj_boxes = boxes[obj_idx]
        obj_labels = labels[obj_idx]
        
        for j in range(obj_scores.shape[0]):

            obj_score = obj_scores[j]
            obj_box = obj_boxes[j].tolist()
            obj_box = list(map(int, obj_box))
            obj_label = obj_labels[j]
            c1 = obj_box[:2]
            c2 = obj_box[2:]
            color = colors[obj_label]
            cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            tf = 1  # font thickness
            class_name = classes[obj_label]

            t_size = cv2.getTextSize(class_name+' '+str(round(obj_score.item(), 2)), 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, class_name+' '+str(round(obj_score.item(), 2)), (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA) # initial 2.35
        
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), frame)
    print('\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SODFormer asynchronous prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.vis_dir:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
    main(args)
