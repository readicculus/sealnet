import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data as data_utils
import numpy as np
from torchvision.transforms import transforms

from utils import get_train_test_meta_data, get_train_test_base


class TrainLoader(Dataset):
    def __init__(self, config, img_size=640, augment=True, multiscale=True, normalized_labels=True):
        try:
            pickle_file = open(config, 'rb')
            config = pickle.load(pickle_file)
        except:
            raise Exception("Could not load file " + config)

        self.train_meta, _ = get_train_test_meta_data(config)

        self.train_path, _ = get_train_test_base(config)
        self.images = list(self.train_meta.keys())
        self.batch_count = 0
        self.img_size = img_size

    def __len__(self):
        return len(self.train_meta)

    def __getitem__(self, idx):
        img = self.images[idx]
        image_path = os.path.join(self.train_path, img)

        hotspots = self.train_meta[img].hotspots
        hs_mat = np.zeros((len(hotspots), 6))
        for i, hs in enumerate(hotspots):
            hs_mat[i] = [0, hs.label[0], hs.x1, hs.y1, hs.x2, hs.y2]

        targets = torch.from_numpy(hs_mat)
        img = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        return image_path, img, targets


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.cat(imgs, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = torch.random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        # imgs = torch.stack([np.resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

def get_train_loader(config, batch_size, num_workers):
    train_dataset = TrainLoader(config)
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers, collate_fn=train_dataset.collate_fn)