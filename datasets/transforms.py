"""
Transforms and data augmentation for both frames, events and boxes.
"""
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh


def crop(image, event, target, region):
    cropped_image = F.crop(image, *region) if image is not None else None
    cropped_event = F.crop(event, *region) if event is not None else None
        
    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels"]

    if "boxes" in target:
        boxes = target["boxes"]
        if boxes.ndim > 1:
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            target["boxes"] = cropped_boxes.reshape(-1, 4)

            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            fields.append("boxes")

            for field in fields:
                target[field] = target[field][keep]

    return cropped_image, cropped_event, target


def hflip(image, event, target):
    flipped_image = F.hflip(image) if image is not None else None
    flipped_event = F.hflip(event) if event is not None else None

    w, h = image.size if image is not None else event.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        if boxes.ndim > 1:
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            target["boxes"] = boxes

    return flipped_image, flipped_event, target


def resize(image, event, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    image_size = image.size if image is not None else event.size
    size = get_size(image_size, size, max_size)
    rescaled_image = F.resize(image, size) if image is not None else None
    rescaled_event = F.resize(event, size) if event is not None else None

    if rescaled_image is not None:
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image_size))
    else:
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_event.size, image_size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        if boxes.ndim > 1:
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, rescaled_event, target


def pad(image, event, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1])) if image is not None else None
    padded_event = F.pad(event, (0, 0, padding[0], padding[1])) if event is not None else None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    return padded_image, padded_event, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, event, target):
        region = T.RandomCrop.get_params(img, self.size) if img is not None else T.RandomCrop.get_params(event, self.size)
        return crop(img, event, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, event, target: dict):
        w_max = min(img.width, self.max_size) if img is not None else min(event.width, self.max_size)
        h_max = min(img.height, self.max_size) if img is not None else min(event.height, self.max_size)
        w = random.randint(self.min_size, w_max)
        h = random.randint(self.min_size, h_max)
        region = T.RandomCrop.get_params(img, [h, w]) if img is not None else T.RandomCrop.get_params(event, [h, w])
        return crop(img, event, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, event, target):
        image_width, image_height = img.size if img is not None else event.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, event, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, event, target):
        if random.random() < self.p:
            return hflip(img, event, target)
        return img, event, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, event, target=None):
        size = random.choice(self.sizes)
        return resize(img, event, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, event, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, event, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, event, target):
        if random.random() < self.p:
            return self.transforms1(img, event, target)
        return self.transforms2(img, event, target)


class ToTensor(object):
    def __call__(self, img, event, target):
        if img is not None and event is not None:
            return F.to_tensor(img), F.to_tensor(event), target
        elif img is None:
            return None, F.to_tensor(event), target
        else:
            return F.to_tensor(img), None, target


class Normalize(object):
    def __init__(self, mean_img, std_img, mean_eve, std_eve):
        self.mean_img = mean_img
        self.std_img = std_img
        self.mean_eve = mean_eve
        self.std_eve = std_eve
    # def __init__(self, mean, std):
    #     self.mean = mean
    #     self.std = std

    def __call__(self, image, event, target=None):
        # image = F.normalize(image, mean=self.mean, std=self.std) if image is not None else None
        image = F.normalize(image, mean=self.mean_img, std=self.std_img) if image is not None else None
        event = F.normalize(event, mean=self.mean_eve, std=self.std_eve) if event is not None else None
        if target is None:
            return image, event, None
        target = target.copy()
        h, w = image.shape[-2:] if image is not None else event.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, event, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, event, target):
        for t in self.transforms:
            image, event, target = t(image, event, target)
        return image, event, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
