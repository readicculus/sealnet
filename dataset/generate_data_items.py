import os
import warnings
from utils import get_git_revisions_hash
import pickle
from data import SealDataset
import argparse

parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)

args = parser.parse_args()

# Load the configuration
config = None
try:
    pickle_file_path = args.config_path
    pickle_file = open(pickle_file_path,'rb')
    config = pickle.load(pickle_file)
except:
    raise Exception("Could not load file " + pickle_file_path)

current_hash = get_git_revisions_hash()
if current_hash != config.hash:
    warnings.warn("Current git hash is not equal to config git hash")


dataset_base = os.path.join(config.generated_data_base, str(config.dataset_id))
train_list = os.path.join(dataset_base, config.system.train_list)
test_list = os.path.join(dataset_base, config.system.test_list)
train_base = os.path.join(dataset_base, "train")
test_base = os.path.join(dataset_base, "test")

# Check required outline files exist
if not os.path.exists(dataset_base):
    raise Exception("specified dataset " + dataset_base + " does not exist.")
if not os.path.isfile(train_list):
    raise Exception("specified train list " + train_list + " does not exist.")
if not os.path.isfile(test_list):
    raise Exception("specified test list " + test_list + " does not exist.")
# create train/test directories
if not os.path.exists(train_base):
    os.makedirs(train_base)
if not os.path.exists(test_base):
    os.makedirs(test_base)

train_dataset = SealDataset(csv_file=train_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')
test_dataset = SealDataset(csv_file=test_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')



print("Loaded test/train files %d / %d" % (len(test_dataset), len(train_dataset)))



