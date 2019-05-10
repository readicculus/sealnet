import argparse
import os
import pickle

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


dataset_base = os.path.join(config.generated_data_base, str(config.dataset_id))
train_list = os.path.join(dataset_base, config.system.train_list)
test_list = os.path.join(dataset_base, config.system.test_list)
train_base = os.path.join(dataset_base, "train")
test_base = os.path.join(dataset_base, "test")


f_tr = open(os.path.join(train_base, "metadata.pickle"), 'rb')
f_te = open(os.path.join(test_base, "metadata.pickle"), 'rb')
train_meta = pickle.load(f_tr)
test_meta = pickle.load(f_te)


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