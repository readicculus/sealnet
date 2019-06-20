import os

from torchvision.transforms import Compose

from transforms.filter_transforms import transform_removed, transform_updated, transform_seal_only
from utils import get_git_revisions_hash, obj

config = obj({
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
        "train_list": "train.csv",
        "test_list": "test.csv"
    }
})