import os

from torchvision.transforms import Compose
from transforms.filter_transforms import *
from utils import get_git_revisions_hash
import configparser



class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


config = {
    "chip_size" : [640,640],
    "dataset": "optical",
    "types": ["optical", "ir", "registered"],
    "generated_data_base": "/data/generated_data/",
    "raw_data_base": "/data/raw_data/",
    "transforms": Compose([
    transform_removed,
    transform_updated,
    transform_seal_only
]),
    "optical_dir": "TrainingAnimals_ColorImages",
    "hash": get_git_revisions_hash(),
    "system": {
        "train_list": "train.txt",
        "test_list": "test.txt"
    }
}


config = obj(config)
