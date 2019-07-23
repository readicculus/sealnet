import argparse
import glob
import os
import pickle
from random import shuffle
from csvloader import LABEL_NAMES
from utils import get_train_test_meta_data, get_train_test_base

parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('--train_background',
                    default=0, help='Number of background images to include in training', type=int)
parser.add_argument('--test_background',
                    default=0, help='Number of background images to include in testing', type=int)
parser.add_argument('-bg_dir', '--background_directory', dest='background_directory', required=False)

args = parser.parse_args()

if not args.background_directory is None:
    if not os.path.isdir(args.background_directory):
        print("Background directory given does not exist: %s" % args.background_directory)
        quit()
if args.train_background == args.test_background == 0 and not args.background_directory is None:
    print("Must give a background directory if # of test or train background chips to include is over 0.")
    quit()

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

background_paths = []
if args.train_background > 0 or args.test_background > 0:
    background_paths = glob.glob(os.path.join(args.background_directory, '*.jpg'))
if args.train_background + args.test_background > len(background_paths):
    print("Not enough background images, in total there are %d images." % len(background_paths))
    quit()

shuffle(background_paths)

train_background_paths = []
test_background_paths = []
if args.train_background > 0:
    train_background_paths = background_paths[:args.train_background]
if args.test_background > 0:
    test_background_paths = background_paths[args.train_background:args.train_background + args.test_background]


def yolo_labels(meta, base):
    unique_classes = set()
    for img_filename in meta:
        txt_file_name = os.path.splitext(img_filename)[0] + ".txt"
        m = meta[img_filename]
        h,w = m.crop.height, m.crop.width
        labels = []
        for hs in m.hotspots:
            center_x = hs.center_x/w
            center_y = hs.center_y/h
            box_w = hs.width/w
            box_h = hs.height/h
            class_id = hs.label[0]
            unique_classes.add(class_id)
            labels.append("%d %.10f %.10f %.10f %.10f" % (class_id, center_x, center_y, box_w, box_h))

        with open(os.path.join(base, txt_file_name), 'w') as f:
            for label in labels:
                f.write("%s\n" % label)
    return unique_classes

def image_list(meta, base, background_images, list_name = "yolo.labels"):
    list_file = os.path.join(base, list_name)
    all_filles = []
    for img_filename in meta:
        full_path = os.path.join(base, img_filename)
        all_filles.append(full_path)
    all_filles = all_filles + background_images

    shuffle(all_filles)

    with open(list_file, 'w') as f:
        for img in all_filles:
            f.write("%s\n" % img)

    return list_file

def gen_data_file(train_list, test_list, classes):
    names_file = os.path.join(config.generated_data_base, str(config.dataset_id),"yolo.names")
    with open(names_file, 'w') as f:
        for c in classes:
            f.write(LABEL_NAMES[c] + "\n")
    data_content = \
        "classes = %d\ntrain  = %s\nvalid = %s\ntest  = %s\nnames  = %s\nbackup  = %s\n" \
        % (len(classes), train_list, test_list, test_list, names_file, "s")
    data_file = os.path.join(config.generated_data_base, str(config.dataset_id),"yolo.data")

    with open(data_file, 'w') as f:
        f.write(data_content)




unique_train = yolo_labels(train_meta, train_base)
unique_test = yolo_labels(test_meta, test_base)
classes = list(unique_test.union(unique_train))


train_list = image_list(train_meta, train_base, train_background_paths)
test_list = image_list(test_meta, test_base, test_background_paths)
gen_data_file(train_list, test_list, classes)