import os

from torchvision.transforms import Compose
import numpy as np
import pandas as pd

import dataset.config as config
from dataset.data import SealDataset
from dataset.transforms.filter_transforms import *
from dataset.utils import test_train_split_by_image
import pickle

config = config.config

filehandler = open("test.pickle", 'wb')
pickle.dump(config, filehandler)
filehandler = open("test.pickle", 'rb')
object = pickle.load(filehandler)

GENERATED_DATASET_BASE = os.path.split("/data/generated_data/")
data_filters = object.transforms

seal_dataset = SealDataset(csv_file='data/TrainingAnimals_WithSightings_updating.csv',
                           root_dir='/data/raw_data/TrainingAnimals_ColorImages/', data_filters=data_filters)



train, test = test_train_split_by_image(seal_dataset, .5)

with open("test.csv", "w") as text_file:
    text_file.write(test.to_csv(index=False))
with open("train.csv", "w") as text_file:
    text_file.write(train.to_csv(index=False))
a=1