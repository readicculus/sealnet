import json
from PIL import Image
from torch.utils.data import Dataset
import requests
import pickle

class ChipLabels(object):
    def __init__(self, chip_region_id, relative_labels):
        self.chip_region_id = chip_region_id
        self.relative_labels = relative_labels

    def add_relative_label(self, relative_labels):
        self.relative_labels.append(relative_labels)


class ImageChips(object):
    def __init__(self, original_labels, intersect_thresh=.4):
        final_chip_ids = set()

        # add one fully intersected chip to the final set
        for k in original_labels:
            for chip in original_labels[k]:
                if chip['percent_intersection'] < 1.0:
                    continue
                final_chip_ids.add(chip['chip_id'])

        res = {x: ChipLabels(x, []) for x in final_chip_ids}
        for k in original_labels:
            for chip in original_labels[k]:
                if chip['chip_id'] in final_chip_ids and chip['percent_intersection'] > intersect_thresh:
                    chip['global_label_id'] = k
                    res[chip['chip_id']].add_relative_label(chip)

        self.chips = [v for k, v in res.items()]


class RemoteChipsDataset(Dataset):
    def __init__ (self, ml_data_type,
                 transform=None, data_filters=None, image_transforms=None,
                 host = "https://www.yuvalboss.com", chip_width=608, chip_height=608, endpoint='/api/hotspots', method='post'):

        self.chip_width = chip_width
        self.chip_height = chip_height
        self.host = host
        self.endpoint = endpoint
        # config for /api/hotspots endpoint
        self.query_config = {
            "species_list": [
                "Polar Bear", "Bearded Seal", "Ringed Seal"
            ],
            "workers": [],
            "jobs": [],
            "surveys": [],
            "flights": [],
            "camera_positions": [],
            "image_type": "eo",
            "show_shadows": False,
            "show_removed_labels": False,
            "ml_data_type": ml_data_type
        }
        HOTSPOTS_ENDPOINT = "%s%s"%(self.host,self.endpoint)
        if method == 'post':
            headers = {'Content-type': 'application/json'}
            r = requests.post(url=HOTSPOTS_ENDPOINT, data=json.dumps(self.query_config), headers=headers)
        else:
            r = requests.get(url=HOTSPOTS_ENDPOINT)

        if r.status_code != 200:
            raise Exception(r.text)
        response = json.loads(r.text)
        self.images = response["images"]
        self.labels_by_image_id = response["labels"]
        self.label_count = response["label_count"]
        self.image_count = response["image_count"]
        self.image_ids = list(self.images.keys())
        self.labels_by_label_id = {}
        self.species_by_id = self.get_species()

        for k in self.labels_by_image_id:
            for label in self.labels_by_image_id[k]:
                self.labels_by_label_id[label['id']] = label

        self.chips_by_image_id, self.chip_regions_by_id = self.chips()

        if data_filters:
            self.data = data_filters(self.data)

        self.transform = transform

    def chips(self):
        label_ids = []
        img_dims = []
        for id in self.image_ids:
            im = self.images[id]
            img_dims.append((im['width'], im['height']))
            im_labels = self.labels_by_image_id[id]
            for label in im_labels:
                label_ids.append(label['id'])
        CHIPS_ENDPOINT = "%s/api/chips?w=%d&h=%d"%(self.host, self.chip_width, self.chip_height)
        headers = {'Content-type': 'application/json'}
        r = requests.post(url=CHIPS_ENDPOINT, data=json.dumps(label_ids), headers=headers)
        if r.status_code != 200:
            raise Exception(r.text)
        response = json.loads(r.text)
        image_chips = response['imageid_to_labelid_to_chipid']
        chip_regions_by_id = response['chips_by_id']
        chips_by_image_id = {}
        for im_id in image_chips:
            c = ImageChips(image_chips[im_id])
            chips_by_image_id[im_id] = c

        return chips_by_image_id, chip_regions_by_id

    def get_species(self):
        SPECIES_ENDPOINT = "%s/api/species"%self.host
        r = requests.get(url=SPECIES_ENDPOINT)
        if r.status_code != 200:
            raise Exception(r.text)
        response = json.loads(r.text)
        return response

    def get_chip_region(self, chip_region_id):
        return self.chip_regions_by_id[str(chip_region_id)]

    def get_label_by_id(self, label_id):
        return self.labels_by_label_id[int(label_id)]

    def get_species_by_id(self, species_id):
        return self.species_by_id[str(species_id)]

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self.images[img_id]
        full_img_path = img['file_path']
        image = None
        try:
            image = Image.open(full_img_path)
        except:
            print("Failed to load: %s" % full_img_path)


        chips = self.chips_by_image_id[img_id] if img_id in self.chips_by_image_id else ImageChips([])

        sample = {'image': {'im_id': img_id, 'im':image, 'im_name':img['file_name'] }, 'chips': chips}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def load(f):
        content = None
        with open(f, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        return content

    def save(self, f):
        with open(f, 'wb') as output:
            pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)
