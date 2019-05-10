import subprocess

import numpy as np
import pandas as pd

# because multiple hotspots can be next to eachother in an image we want to split our train/test
# by image.  There are probably other reasons this is better too...
def test_train_split_by_image(dataset, train_ratio = .8):
    df = pd.DataFrame(np.random.randn(len(dataset.images), 2))
    msk = np.random.rand(len(df)) < train_ratio

    train_images = dataset.images[msk]
    test_images = dataset.images[~msk]

    train = dataset.data[dataset.data.color_image.isin(train_images)]
    test = dataset.data[dataset.data.color_image.isin(test_images)]
    return train, test

def get_git_revisions_hash():
     return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)