import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from dataset.data import SealDataset
from dataset.transforms.crops import get_tile_images


# Create a dataset of chips with respective label files

seal_dataset = SealDataset(csv_file='../data/updated_seals.csv',
                                    root_dir='/data/raw_data/TrainingAnimals_ColorImages/')

CHIP_SHAPE = 640

buffer = []
for i in range(len(seal_dataset)):
    sample = seal_dataset[i]

    full_res = sample["image"]
    shape = (full_res.height, full_res.width, full_res.layers)
    labels = np.array(sample["labels"])
    boxes = np.array(sample["boxes"])
    bboxes = []
    for idx, box in enumerate(boxes):
        center_x = box[0] + (box[2] - box[0])/2
        center_y = box[1] + (box[3] - box[1]) / 2
        t = ia.BoundingBox(box[0],box[1], box[2], box[3], label =labels[idx])
        bboxes.append(t)

    bbs = ia.BoundingBoxesOnImage(bboxes, shape=shape)

    aug = iaa.CropToFixedSize(640,640)
    a = aug.augment_bounding_boxes(bbs)
    for bb in a.bounding_boxes:
        print(bb.is_fully_within_image(a))
    # for box in bboxes:
    #     crop_box = ia.BoundingBox(box.center_x - CHIP_SHAPE/2, box.center_y - CHIP_SHAPE/2,
    #                               box.center_x + CHIP_SHAPE / 2, box.center_y + CHIP_SHAPE / 2)
    #     for box_i in bboxes:
    #         crop_box.contains(box_i)


    x=1
