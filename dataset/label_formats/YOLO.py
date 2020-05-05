import os

from label_formats.LabelFormatter import LabelFormatter


class YOLO(LabelFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.species = []
        self.train_im = [pos_json for pos_json in os.listdir(self.train_dir) if pos_json.endswith('.JPG')]
        self.test_im = [pos_json for pos_json in os.listdir(self.test_dir) if pos_json.endswith('.JPG')]

    def generate_metadata_files(self):
        names_file = os.path.join(self.dataset_dir, "yolo.names")
        data_file = os.path.join(self.dataset_dir, "yolo.data")
        train_list_file = os.path.join(self.dataset_dir, "train", "yolo.labels")
        test_list_file = os.path.join(self.dataset_dir, "test", "yolo.labels")
        backup_dir = os.path.join(self.dataset_dir, "weights")

        # create weights directory
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # write .data file
        data_content = \
            "classes = %d\ntrain  = %s\nvalid = %s\ntest  = %s\nnames  = %s\nbackup  = %s\n" \
            % (len(self.species), train_list_file, test_list_file, test_list_file, names_file, backup_dir)
        with open(data_file, 'w') as f:
            f.write(data_content)

        # write .names file
        with open(names_file, 'w') as f:
            for s in self.species:
                f.write('%s\n' % s)

        # write test and train list files
        with open(train_list_file, 'w') as f:
            for s in self.train_im:
                fp = os.path.join(self.train_dir, s)
                f.write('%s\n' % fp)

        with open(test_list_file, 'w') as f:
            for s in self.test_im:
                fp = os.path.join(self.test_dir, s)
                f.write('%s\n' % fp)

    def generate_labels(self, dir, json_files):
        for f in json_files:
            fp = os.path.join(dir, f)
            labels = self.load_json(fp)
            lines = []
            for label in labels:
                global_label=label['global_label']
                relative_label=label['relative_label']
                species_name=label['species_name']
                if not species_name in self.species:
                    self.species.append(species_name)

                im_w=label['im_w']
                im_h=label['im_h']
                x1 = relative_label['relative_x1']
                x2 = relative_label['relative_x2']
                y1 = relative_label['relative_y1']
                y2 = relative_label['relative_y2']

                class_id = self.species.index(species_name)
                x_center = (x1 + (x2-x1)/2.0)/im_w
                y_center = (y1 + (y2-y1)/2.0)/im_h
                w = (x2-x1)/im_w
                h = (y2-y1)/im_h
                lines.append("%d %f %f %f %f" % (class_id, x_center, y_center, w, h))

            label_fp = fp.replace('.json', '.txt')
            with open(label_fp, 'w') as text_file:
                for line in lines:
                    text_file.write(line + '\n')

lf = YOLO("/fast/generated_data/dataset_1/")
lf.format()