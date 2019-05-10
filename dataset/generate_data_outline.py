import os

import config as config
from csvloader import SealDataset
from utils import test_train_split_by_image, get_git_revisions_hash
import pickle

# Generate the outline of a dataset.  What does this mean?
# Essentially its nicer to break this up into two steps.  In this step we create the new folder in which
# the new dataset will reside, we create a test/train split based on the config, and we pickle the config file.
# The next step is generate_data_items which actually does the chipping based on the outlined files and config
# that we generate here.
config = config.config

raw_image_path = os.path.join(config.raw_data_base, config.optical_dir)

dataset_id = len(os.listdir(config.generated_data_base))
dataset_base = os.path.join(config.generated_data_base, str(dataset_id))
config.dataset_id = dataset_id

if not os.path.exists(dataset_base):
    os.makedirs(dataset_base)
else:
    raise Exception('directory %s already exists' % dataset_base)

filehandler = open(os.path.join(dataset_base,"config.pickle"), 'wb')
pickle.dump(config, filehandler)


seal_dataset = SealDataset(csv_file='data/TrainingAnimals_WithSightings_updating.csv',
                           root_dir='/data/raw_data/TrainingAnimals_ColorImages/', data_filters=config.transforms)

train, test = test_train_split_by_image(seal_dataset, .5)

with open(os.path.join(dataset_base,config.system.test_list), "w") as text_file:
    text_file.write(test.to_csv(index=False))
with open(os.path.join(dataset_base,config.system.train_list), "w") as text_file:
    text_file.write(train.to_csv(index=False))

print("Created new dataset outline in: %s" % dataset_base)

