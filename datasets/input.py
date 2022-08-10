"""
PKU-DAVIS-SOD dataset which returns synchronous streams of event and frame for training and evaluation.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import Dataset
import datasets.transforms as T


class DVSDetection(Dataset): # load frames and events in a video
    def __init__(self, aps_dir, ann_folder, npy_dir, no_frame, no_event, event_repre, transforms, drop_last, scene, batch_size):
        self._transforms = transforms
        self.prepare = Prepare('dvs')
        self.aps_dir = aps_dir
        self.anno = ann_folder # path to the folder of the annotation files for the dvs file
        self.npy_dir = npy_dir
        assert os.path.exists(aps_dir) and os.path.exists(ann_folder) and os.path.exists(npy_dir)
        self.no_frame = no_frame
        self.no_event = no_event
        self.event_repre = event_repre
        self.drop_last = drop_last
        self.scene = scene
        self.frame_len = len(os.listdir(self.aps_dir))
        self.events_len = len(os.listdir(self.npy_dir))

        assert not (self.no_frame and self.no_event)

        if not self.no_frame:
            if drop_last:
                drop_index = self.frame_len % batch_size
                self.frame_len -= drop_index
            
        if not self.no_event:
            if drop_last:
                drop_index = self.events_len % batch_size
                self.events_len -= drop_index

        if not self.no_frame and not self.no_event:
            assert self.frame_len == self.events_len

    def __getitem__(self, idx):
        clses, bboxes = get_json_boxes(os.path.join(self.anno, '{}.json'.format(idx + 1)))
        target = {'image_id': idx, 'boxes': bboxes, 'labels': clses}

        if not self.no_frame:
            img = Image.open(os.path.join(self.aps_dir, '{}.png'.format(idx)))
        else:
            img = None

        if not self.no_event:
            event = np.load(os.path.join(self.npy_dir, '{}.npy'.format(idx)))
            if self.event_repre == 'image':
                event = make_color_histo(event, width=346, height=260)
            elif self.event_repre == 'voxel':
                event = events_to_voxel_grid(event, 3, 260, 346).astype(np.uint8)
            event = Image.fromarray(event)
        else:
            event = None

        target = self.prepare(img, event, target)

        img, event, target = self._transforms(img, event, target)
        return img, event, target

    def __len__(self):
        if self.no_frame:
            return self.events_len
        else:
            return self.frame_len


class Prepare(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, image, event, target):
        if image:
            w, h = image.size
        else:
            w, h = event.size

        gt = {}
        gt["orig_size"] = torch.as_tensor([int(h), int(w)])
        gt["size"] = torch.as_tensor([int(h), int(w)])

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        if self.dataset == 'dvs':
            boxes = target['boxes']
            classes = target['labels']
        else:
            anno = target["annotations"]
            anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            boxes = [obj["bbox"] for obj in anno]
            classes = [obj["category_id"] for obj in anno]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.dataset == 'coco':
            boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
    
        gt["boxes"] = boxes
        gt["labels"] = classes
        gt["image_id"] = image_id
        
        return gt


def get_json_boxes(label_filename):
    if not os.path.exists(label_filename):
        return [], []

    with open(label_filename, 'r') as json_file:
        data = json.load(json_file)
        objects = data['shapes']
        class_indexes = []
        bounding_boxes = []
        for i in range(len(objects)):
            bounding_boxes_points = objects[i]['points']
            if 'label' in objects[i]:
                bounding_boxes_class = objects[i]['label']
            else:
                bounding_boxes_class = objects[i]['lable']
                
            class_index = int(bounding_boxes_class)
            bounding_box = [int(bounding_boxes_points[0][0]), int(bounding_boxes_points[0][1]), int(bounding_boxes_points[1][0]), int(bounding_boxes_points[1][1])]

            class_indexes.append(class_index)
            bounding_boxes.append(bounding_box)

    return class_indexes, bounding_boxes


def events_to_voxel_grid(events, num_bins, height, width):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    if events.size == 0:
        voxel_grid = np.zeros((height, width, num_bins), np.uint8)
        return voxel_grid

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1]['timestamp']
    first_stamp = events[0]['timestamp']
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events['timestamp'] = (num_bins - 1) * (events['timestamp'] - first_stamp) / deltaT
    ts = events['timestamp']
    xs = events['x'].astype(np.int)
    ys = events['y'].astype(np.int)
    pols = events['polarity']
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    voxel_grid = voxel_grid.transpose(1, 2, 0)

    return voxel_grid


def make_color_histo(events, img=None, width=346, height=260):
    """
    simple display function that shows negative events as blue dots and positive as red one
    on a white background
    args :
        - events structured numpy array: timestamp, x, y, polarity.
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int.
        - height int.
    return:
        - img numpy array, height x width x 3.
    """
    if img is None:
        img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 255
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        ON_index = np.where(events['polarity'] == 1)

        img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['polarity'][ON_index][:, None]  # red

        OFF_index = np.where(events['polarity'] == 0)
        img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['polarity'][OFF_index] + 1)[:,None]  # blue
    
    return img


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                    [0.927, 0.856, 0.899], [0.147, 0.223, 0.183])
    ])

    scales = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576]
    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=600),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([352], max_size=600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, seq_scene, seq_id):
    png_root = Path(args.frame_path)
    ann_root = Path(args.anno_path)
    npy_root = Path(args.event_path)
    PATHS_dvs = {
    "train": (png_root / "train/{}/{}".format(seq_scene, seq_id),
              ann_root / "train/{}/{}".format(seq_scene, seq_id),
              npy_root / "train/{}/{}".format(seq_scene, seq_id)),
    "val": (png_root / "val/{}/{}".format(seq_scene, seq_id),
            ann_root / "val/{}/{}".format(seq_scene, seq_id),
            npy_root / "val/{}/{}".format(seq_scene, seq_id)),
    "test": (png_root / "test/{}/{}".format(seq_scene, seq_id),
             ann_root / "test/{}/{}".format(seq_scene, seq_id),
             npy_root / "test/{}/{}".format(seq_scene, seq_id))
    }
    aps_dir, ann_folder, npy_dir = PATHS_dvs[image_set]
    drop_last = False if args.no_temporal else True
    dataset = DVSDetection(aps_dir, ann_folder, npy_dir, no_frame=args.no_frame, no_event=args.no_event,
                           event_repre=args.event_repre, transforms=make_transforms(image_set), drop_last=drop_last,
                           scene=seq_scene, batch_size=args.batch_size)
    return dataset
