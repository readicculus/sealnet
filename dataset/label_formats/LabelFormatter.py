import json
import os


class LabelFormatter(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train/")
        self.test_dir = os.path.join(self.dataset_dir, "test/")

        self.train_json = [pos_json for pos_json in os.listdir(self.train_dir) if pos_json.endswith('.json')]
        self.test_json = [pos_json for pos_json in os.listdir(self.test_dir) if pos_json.endswith('.json')]

    def generate_metadata_files(self):
        pass

    def generate_labels(self, dir, json_files):
        pass

    def load_json(self, fp):
        data = None
        with open(fp) as json_file:
            data = json.load(json_file)
        return data

    def format(self):
        self.generate_labels(self.train_dir, self.train_json)
        self.generate_labels(self.test_dir, self.test_json)
        self.generate_metadata_files()

