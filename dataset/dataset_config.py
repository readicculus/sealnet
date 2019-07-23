import os

from torchvision.transforms import Compose

from transforms.filter_transforms import *
from utils import get_git_revisions_hash, obj

config = obj({
    "chip_size" : [416,416],
    "dataset": "optical",
    "types": ["optical", "ir", "registered"],
    "generated_data_base": "/fast/generated_data/",
    "raw_data_base": "/data/raw_data/",
    "transforms": Compose([
    transform_removed,
    transform_updated,
    transform_seal_only,
    transform_remove_unk_seal,
    transform_remove_bad_res,
    transform_remove_off_edge
    ]),
    "image_augmentations": Compose([
    ]),
    "optical_dir": "TrainingAnimals_ColorImages",
    "hash": get_git_revisions_hash(),
    "system": {
        "train_list": "train.csv",
        "test_list": "test.csv"
    },
    "description": ""
})