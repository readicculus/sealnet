from dataset.crops import get_tile_images
from data import SealDataset


import numpy as np

def tranform_seal_only(data):
    # remove hotspots that I labeled "removed"
    remove_removed = data[~data.status.str.contains("removed")]
    updated_only = remove_removed[remove_removed.updated == True]
    seal_only = updated_only[updated_only.species_id.str.contains("Seal")]
    return seal_only

seal_dataset = SealDataset(csv_file='data/updated_seals.csv',
                                    root_dir='/data/raw_data/TrainingAnimals_ColorImages/', data_filter=tranform_seal_only)
# with open("data/updated_seals.csv", "w") as text_file:
#     text_file.write(seal_dataset.updated_seals.to_csv(index=False))
dims = 640


buffer = []
for i in range(len(seal_dataset)):
    sample = seal_dataset[i]

    full_res = np.array(sample["image"])
    boxes = np.array(sample["boxes"])
    for box in boxes:
        center_x = box[0] + (box[2] - box[0])/2
        center_y = box[1] + (box[3] - box[1]) / 2
    img_shape = full_res.shape

    labels = np.zeros(img_shape)
    buffer = buffer+get_tile_images(full_res)

    x=1
