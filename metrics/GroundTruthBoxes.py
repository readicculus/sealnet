import pandas as pd
import pickle
import os
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *

YUVALS_CSV = True
pkl_file = "metrics/pickels/ground_truth_bounding_boxes.pkl"

# function to generate a BoundingBoxes object containing ground truth labels from the ground truth csv file\
# saves the object as a pickel file for faster loading
def get_ground_truth_boxes():
    if os.path.exists(pkl_file):
        return pickle.load(open(pkl_file, "rb"))
    ground_truth_data = pd.read_csv("/home/yuval/Documents/XNOR/sealnet/data/updated_seals.csv", dtype={'hotspot_id': object})
    bounding_boxes = BoundingBoxes()
    x1_col, x2_col, y1_col, y2_col = "color_left","color_right","color_top","color_bottom"  # Use for original NOAA format
    numeric_cols = ["thermal_x", "thermal_y", "color_left", "color_top",
                    "color_right", "color_bottom"]

    if YUVALS_CSV:
        x1_col, x2_col, y1_col, y2_col = "updated_left", "updated_right", "updated_top", "updated_bottom"  # Use for files w/my udpated labels

        numeric_cols = ["thermal_x", "thermal_y", "color_left", "color_top",
                        "color_right", "color_bottom", "updated_left",
                        "updated_top", "updated_right", "updated_bottom"]
    ground_truth_data[numeric_cols] = \
        ground_truth_data[numeric_cols].apply(pd.to_numeric)

    for index, row in ground_truth_data.iterrows():
        hsId = row["hotspot_id"]
        x1 = row[x1_col]
        x2 = row[x2_col]
        y1 = row[y1_col]
        y2 = row[y2_col]
        label = row['species_id']
        img_name = row['color_image']
        bbox = BoundingBox(imageName=img_name, classId=label,
                           x=x1, y=y1, w=x2, h=y2, typeCoordinates=CoordinatesType.Absolute,
                           bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, hsId=hsId
                           )
        bounding_boxes.addBoundingBox(bbox)

    pickle.dump(bounding_boxes, open(pkl_file, "wb"))
    return bounding_boxes