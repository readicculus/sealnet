import argparse
import imghdr
import os
import pickle

from PIL import Image
import numpy as np
from csvloader import LABEL_NAMES
from utils import get_train_test_meta_data, get_train_test_base
import tensorflow as tf
import contextlib2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
import cv2
from scipy.ndimage import imread

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

def create_tf_record(m, img_filename, encoded_image_data):
    image_format = b'jpeg'
    im_height, im_width = m.crop.height, m.crop.width
    labels = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for hs in m.hotspots:
        class_id = hs.label[0]
        xmins.append(hs.x1 / im_width)
        ymins.append(hs.y1 / im_height)
        xmaxs.append(hs.x2 / im_width)
        ymaxs.append(hs.y2 / im_height)
        labels.append(class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(im_height.astype(int)),
        'image/width': dataset_util.int64_feature(im_width.astype(int)),
        'image/filename': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(labels),
    }))
    return tf_example
def yolo_labels(meta, base):
    unique_classes = set()
    num_shards = 100
    output_filebase = os.path.join(base, 'tf.record')
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)

        for index, img_filename in enumerate(meta):
            img_path = os.path.join(base, img_filename)
            # image = Image.open(img_path)
            # image = np.asarray(image, np.uint8)
            # encoded = image.tobytes()
            # image = cv2.imread(img_path)
            # file_type = imghdr.what(img_path)
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()
            m = meta[img_filename]
            output_shard_index = index % num_shards
            record = create_tf_record(m, img_path, image_data)
            output_tfrecords[output_shard_index].write(record.SerializeToString())



    return unique_classes






unique_train = yolo_labels(train_meta, train_base)
unique_test = yolo_labels(test_meta, test_base)
