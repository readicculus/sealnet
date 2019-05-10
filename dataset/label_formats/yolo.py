import argparse
import os
import pickle

from utils import get_train_test_meta_data, get_train_test_base

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

train_base, test_base = get_train_test_base(config)
train_meta, test_meta = get_train_test_meta_data(config)

def yolo_labels(meta, base):
    for image in meta:
        txt_file_name = os.path.splitext(image)[0] + ".txt"
        m = meta[image]
        h,w = m.crop.height, m.crop.width
        labels = []
        for hs in m.hotspots:
            center_x = hs.center_x/w
            center_y = hs.center_y/h
            box_w = hs.width/w
            box_h = hs.height/h
            class_id = hs.label[0]
            labels.append("%d %f %f %f %f" % (class_id, center_x, center_y, box_w, box_h))

        with open(os.path.join(base, txt_file_name), 'w') as f:
            for label in labels:
                f.write("%s\n" % label)

yolo_labels(train_meta, train_base)
yolo_labels(test_meta, test_base)