import os

from torchvision.transforms import Compose

from transforms.filter_transforms import *
from utils import get_git_revisions_hash, obj

config = obj({
    "name": "PB-S",  # PB-S for polar bear and seal, PB for only polar bear, S for only seal
    "chip_size" : [640,640],
    "dataset": "optical",
    "types": ["optical", "ir", "registered"],
    "generated_data_base": "/fast/generated_data/",
    "raw_data_base": "/data/raw_data/",
    "transforms": Compose([
    transform_removed,
    transform_updated,
    transform_seal_and_pb_only,
    transform_remove_unk_seal,
    transform_remove_bad_res,
    transform_remove_off_edge,
    transform_remove_shadow_annotations
    ]),
    "image_augmentations": Compose([
    ]),
    "optical_dir": "TrainingAnimals_ColorImages",
    "hash": get_git_revisions_hash(),
    "system": {
        "train_list": "train.csv",
        "test_list": "test.csv"
    },
    "description": "Polar Bear and Seal"
})