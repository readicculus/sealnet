import os
from PIL import Image
from torch.utils import data as data_utils
from torch.utils.data import Dataset
from torchvision import datasets as torch_datasets
from torchvision import transforms
import pandas as pd

class SealDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        numeric_cols = ["thermal_x", "thermal_y", "color_left", "color_top",
                          "color_right", "color_bottom", "updated_left",
                          "updated_top", "updated_right", "updated_bottom"]
        self.hotspot_data = pd.read_csv(csv_file)

        # remove hotspots that I labeled "removed"
        self.hotspot_data = self.hotspot_data[~self.hotspot_data.status.str.contains("removed")]
        # cast numeric columns
        self.hotspot_data[numeric_cols] = \
            self.hotspot_data[numeric_cols].apply(pd.to_numeric)

        self.updated_hotspots = self.hotspot_data[self.hotspot_data.updated == True]

        self.updated_seals = self.updated_hotspots[self.updated_hotspots.species_id.str.contains("Seal")]

        self.images = self.updated_seals.color_image.unique()

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

        hotspots = self.updated_hotspots[self.updated_hotspots.color_image == img_base_name]
        bboxes = []
        labels = []
        for index, hs in hotspots.iterrows():
            bboxes.append([hs.updated_left, hs.updated_top, hs.updated_right, hs.updated_bottom])
            labels.append(hs.species_id)
        print(len(hotspots))

        sample = {'image': image, 'labels': labels, 'boxes': bboxes}

        if self.transform:
            sample = self.transform(sample)

        return sample

seal_dataset = SealDataset(csv_file='data/TrainingAnimals_WithSightings_updating.csv',
                                    root_dir='/data/raw_data/TrainingAnimals_ColorImages/')

for i in range(len(seal_dataset)):
    sample = seal_dataset[i]
    x = 1