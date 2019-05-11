import argparse
import os
import pickle

from utils import get_train_test_meta_data

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

train_meta, test_meta = get_train_test_meta_data(config)

def print_overview(meta, type):
    print("%s set overview -----------------------------------------" % type)
    class_distribution = {}
    total = 0
    for image in meta:
        m = meta[image]
        for hs in m.hotspots:
            class_id = int(hs.label[0])
            if not class_id in class_distribution:
                class_distribution[class_id] = 0
            class_distribution[class_id] += 1
            total += 1
    print("Total crops: %d" % len(meta))
    print("Total hotspots: %d" % total)
    print("Density: %.2f" % (total/len(meta)))
    for c in class_distribution:
        count = class_distribution[c]
        print("%d: %d - %.2f%%" %(c, count, count/total * 100))


print_overview(train_meta, "Train")
print_overview(test_meta, "Test")
