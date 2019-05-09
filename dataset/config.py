import os

from torchvision.transforms import Compose
from dataset.transforms.filter_transforms import *

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
    "generated_data_base": "",
    "raw_data_base": "/data/raw_data/",
    "transforms": Compose([
    transform_removed,
    transform_updated,
    transform_seal_only
])

}


config = obj(config)