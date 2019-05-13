import os
import pickle
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data as data_utils
import numpy as np
from torchvision.transforms import transforms
import torch.nn.functional as F

import dataset.utils as utils
import dataset.transforms as transformsx
from models.yolov3.utils.augmentations import horisontal_flip

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class TrainLoader(Dataset):
    def __init__(self, config, img_size=640, augment=True, multiscale=True, normalized_labels=True):
        try:
            sys.modules['utils'] = utils
            sys.modules['transforms'] = transformsx
            pickle_file = open(config, 'rb')
            config = pickle.load(pickle_file)
        except:
            raise Exception("Could not load file " + config)

        self.train_meta, _ = utils.get_train_test_meta_data(config)

        self.train_path, _ = utils.get_train_test_base(config)
        self.images = list(self.train_meta.keys())
        self.batch_count = 0
        self.img_size = img_size
        self.normalized_labels = normalized_labels
        self.augment = augment
        self.size = len(self.train_meta)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.train_path, img_name)
        img = transforms.ToTensor()(Image.open(image_path).convert('RGB')).float()
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        hotspots = self.train_meta[img_name].hotspots
        hs_mat = np.zeros((len(hotspots), 6))
        for i, hs in enumerate(hotspots):
            cx = ((hs.x1 + hs.x2)/2)/padded_w
            cy = ((hs.y1 + hs.y2)/2)/padded_h
            w = (hs.x2 - hs.x1)/padded_w
            h = (hs.y2 - hs.y1)/padded_h
            hs_mat[i] = [0, hs.label[0], cx, cy, w,h]

        targets = torch.from_numpy(hs_mat).float()
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return image_path, img, targets


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # imgs = torch.cat(imgs, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = torch.random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

def get_train_loader(config, batch_size, num_workers):
    train_dataset = TrainLoader(config)
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers, collate_fn=train_dataset.collate_fn)