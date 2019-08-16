import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

LABEL_NAMES = ["Ringed Seal", "Bearded Seal", "UNK Seal", "Polar Bear"]
class SealDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, data_filters=None, image_transforms=None):
        self.current_sample = None
        self.label_names = LABEL_NAMES

        numeric_cols = ["thermal_x", "thermal_y", "color_left", "color_top",
                          "color_right", "color_bottom", "updated_left",
                          "updated_top", "updated_right", "updated_bottom"]
        self.data = pd.read_csv(csv_file, dtype={'hotspot_id': object})
        # cast numeric columns
        self.data[numeric_cols] = \
            self.data[numeric_cols].apply(pd.to_numeric)

        if data_filters:
            self.data = data_filters(self.data)


        self.images = self.data.color_image.unique()

        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_base_name = self.images[idx]
        full_img_path = os.path.join(self.root_dir, img_base_name)
        image = None
        try:
            image = Image.open(full_img_path)
        except:
            print("Failed to load: %s" % full_img_path)

        hotspots = self.data[self.data.color_image == img_base_name]

        boxes = []
        labels = []
        hs_ids = []
        for index, hs in hotspots.iterrows():
            boxes.append([hs.updated_left, hs.updated_top, hs.updated_right, hs.updated_bottom])
            labels.append((self.get_label(hs.species_id), hs.hotspot_id))


        sample = {'image':image, 'labels': labels, 'boxes': boxes}

        if self.transform:
            sample = self.transform(sample)

        self.current_sample = sample
        return sample


    def get_label(self, name):
        return self.label_names.index(name)

