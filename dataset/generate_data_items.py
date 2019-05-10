import os
import numpy as np
import warnings

from PIL import Image

from transforms.crops import crop_around_hotspots
from utils import get_git_revisions_hash
import pickle
from csvloader import SealDataset
import argparse
from imgaug import BoundingBox, BoundingBoxesOnImage
from utils import obj

parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('-d', '--debug',  default=False, action='store_true')

args = parser.parse_args()
DEBUG = args.debug

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
    raise Exception("specified dataset {} does not exist.".format(dataset_base))
if not os.path.isfile(train_list):
    raise Exception("specified train list {} does not exist.".format(train_list))
if not os.path.isfile(test_list):
    raise Exception("specified test list {} does not exist.".format(test_list))

# create train/test directories
if os.path.exists(train_base):
    warnings.warn("train directory {} already exists.".format(train_base))
if os.path.exists(test_base):
    warnings.warn("test directory {} already exists.".format(test_base))
if not os.path.exists(train_base):
    os.makedirs(train_base)
if not os.path.exists(test_base):
    os.makedirs(test_base)

# load train and test dataset
train_dataset = SealDataset(csv_file=train_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')
test_dataset = SealDataset(csv_file=test_list, root_dir='/data/raw_data/TrainingAnimals_ColorImages/')

def process_and_create_dataset(dataset, base, type):
    print("Generating %s set--------------------" % type)
    metadata = {}
    for i, hs in enumerate(dataset):
        print("%.3f%% complete"%(i/len(dataset) * 100), sep='', end='\r', flush=True)
        labels = hs["labels"]
        boxes = hs["boxes"]
        image = hs["image"]
        image = np.asarray(image)

        bb = []
        for box, label in zip(boxes, labels):
            bb.append(BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label))
        bbs = BoundingBoxesOnImage(bb, shape=image.shape)
        chips = crop_around_hotspots(bbs, config.chip_size)

        for chip_idx, chip in enumerate(chips):
            crop = chip[0]
            hotspots = chip[1]
            cropped_image = image[int(crop.y1): int(crop.y2),
                            int(crop.x1):int(crop.x2)].astype(np.uint8)
            new_bbs = []
            for hs in hotspots:
                new_bb = hs.shift(left=-crop.x1, top = -crop.y1).clip_out_of_image(cropped_image.shape)
                new_bbs.append(new_bb)


            chip_metadata = obj({"crop": crop, "hotspots": new_bbs})
            file_name = "%d-%d.jpg"%(i,chip_idx)
            metadata[file_name] = chip_metadata
            if DEBUG:
                for bb in new_bbs:
                    cropped_image = bb.draw_on_image(cropped_image, size=10, color=[255, 0, 0])
            img = Image.fromarray(cropped_image, 'RGB')
            img.save(os.path.join(base, file_name))

    filehandler = open(os.path.join(base, "metadata.pickle"), 'wb')
    pickle.dump(metadata, filehandler)


process_and_create_dataset(train_dataset, train_base, "Train")
process_and_create_dataset(test_dataset, test_base, "Test")
print("Loaded test/train files %d / %d" % (len(test_dataset), len(train_dataset)))



