import json
import os.path

from PIL import ImageDraw, ImageFont
from hurry.filesize import size

from RemoteDataloader import RemoteChipsDataset


class ChipPreprocessor(object):
    def __init__(self, dataset_dir, DRAW_LABELS=False):
        self.dataset_dir = dataset_dir
        self.train_dataloader = RemoteChipsDataset.load(os.path.join(self.dataset_dir, 'train.pkl'))
        self.test_dataloader = RemoteChipsDataset.load(os.path.join(self.dataset_dir, 'test.pkl'))
        self.train_dir = os.path.join(self.dataset_dir, "train/")
        self.test_dir = os.path.join(self.dataset_dir, "test/")
        self.DRAW_LABELS = DRAW_LABELS

    def create_test_train_dirs(self):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def im_chip_filename(self,im_name, x1, y1, x2, y2):
        fn = os.path.splitext(im_name)[0]
        return "%s--%d_%d-%d_%d" % (fn, x1, y1, x2, y2)

    def draw_labels(self, chip, relative_labels, dataset):
        for rl in relative_labels:
            gl = dataset.get_label_by_id(rl['global_label_id'])
            species = dataset.get_species_by_id(gl['species_id'])
            x1 = rl['relative_x1']
            x2 = rl['relative_x2']
            y1 = rl['relative_y1']
            y2 = rl['relative_y2']
            img = ImageDraw.Draw(chip)
            img.rectangle([(x1, y1), (x2,y2)], fill=None, outline="red")
            img.text((x1, y1), species, font=ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Regular.ttf", 9))

    def save_label(self, area, relative_labels, dataset, fp):
        res = []
        for rl in relative_labels:
            width = area[2]-area[0]
            height = area[3]-area[1]
            gl = dataset.get_label_by_id(rl['global_label_id'])
            species = dataset.get_species_by_id(gl['species_id'])
            rl['relative_x1'] = max(rl['relative_x1'],0)
            rl['relative_x2'] = min(rl['relative_x2'],width)
            rl['relative_y1'] = max(rl['relative_y1'],0)
            rl['relative_y2'] = min(rl['relative_y2'],height)
            label_dict = {"global_label": gl, "relative_label": rl, "species_name": species, "im_w":width, "im_h":height}
            res.append(label_dict)
        with open(fp, 'w') as text_file:
            text_file.write(json.dumps(res))

    def save_labels(self, chip_dataset, path):
        print("Saving labels to: %s" % path)
        for x in chip_dataset:
            im_name = x['image']['im_name']
            chips = x['chips'].chips
            for chip in chips:
                region = chip_dataset.get_chip_region(chip.chip_region_id)
                area = (region['x1'], region['y1'], region['x2'], region['y2'])
                fn = self.im_chip_filename(im_name, region['x1'], region['y1'], region['x2'], region['y2'])
                json_label_fp = os.path.join(path, "%s.json" % fn)
                self.save_label(area, chip.relative_labels, chip_dataset, json_label_fp)


    def save_chips(self, chip_dataset, path):
        print("Saving chips to: %s" % path)
        total = len(chip_dataset)
        bytes_saved = 0
        i=0
        for x in chip_dataset:
            im = x['image']['im']
            im_name = x['image']['im_name']
            chips = x['chips'].chips

            for chip in chips:
                region = chip_dataset.get_chip_region(chip.chip_region_id)
                area = (region['x1'], region['y1'], region['x2'], region['y2'])
                chip_im = im.crop(area)
                if self.DRAW_LABELS: self.draw_labels(chip_im, chip.relative_labels, chip_dataset)
                fn = self.im_chip_filename(im_name, region['x1'], region['y1'], region['x2'], region['y2'])
                chip_fp = os.path.join(path, "%s.JPG" % fn)
                chip_im.save(chip_fp, quality=100)
                bytes_saved += os.stat(chip_fp).st_size

            if i % 50 == 0:
                print("%d/%d images processed" % (i, total))
                print("byte_saved:%s" % size(bytes_saved))
            i+=1

    def preprocess(self, chip=True, label=True):
        self.create_test_train_dirs()
        if chip:
            self.save_chips(self.train_dataloader, self.train_dir)
            self.save_chips(self.test_dataloader, self.test_dir)
        if label:
            self.save_labels(self.train_dataloader, self.train_dir)
            self.save_labels(self.test_dataloader, self.test_dir)

p = ChipPreprocessor('/fast/generated_data/dataset_1', DRAW_LABELS=False)

p.preprocess(chip=True)