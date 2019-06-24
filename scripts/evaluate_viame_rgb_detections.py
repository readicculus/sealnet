import os
import pandas as pd

from metrics.BoundingBox import *
from metrics.BoundingBoxes import BoundingBoxes
from metrics.Evaluator import Evaluator
from metrics.utils import CoordinatesType

import argparse
# An example detection results file can be found at
# https://drive.google.com/file/d/18PzWcAUwu9kdagB1Cz4XmH5r6kJdvx2p/view?usp=sharing
# which can be evaluated against data/updated_seals.csv in this repo
parser = argparse.ArgumentParser(description='Evaluate RGB Detectors.')
parser.add_argument('--gts', help='Ground Truth CSV Path', required=True)
parser.add_argument('--dets', help='Viame Detections CSV Path', required=True)
parser.add_argument('--nms', default=.5, help='nms threshold', type=float)
parser.add_argument('--iou', default=.5, help='iou threshold', type=float)
parser.add_argument('--conf', default=0.0, help='minimum confidence threshold', type=float)
parser.add_argument('--detectiononly', dest='detectiononly', action='store_true', help='minimum confidence threshold')

args = parser.parse_args()

DETECTIONS_CSV = args.dets
GROUND_TRUTH_CSV = args.gts
NMS_THRESH = args.nms
IOUThreshold = args.iou
CONFIDENCE_THRESH = args.conf
DETECTION_ONLY = args.detectiononly


x1_col, x2_col, y1_col, y2_col = "updated_left","updated_right","updated_top","updated_bottom"  # Use for files w/my udpated labels
# x1_col, x2_col, y1_col, y2_col = "color_left","color_right","color_top","color_bottom"  # Use for original NOAA format

# READ DATA FROM BOTH FILES INTO PANDAS
ground_truth_data = pd.read_csv(GROUND_TRUTH_CSV, dtype={'hotspot_id': object})
numeric_cols = ["thermal_x", "thermal_y", "color_left", "color_top",
                "color_right", "color_bottom", "updated_left",
                "updated_top", "updated_right", "updated_bottom"]
ground_truth_data[numeric_cols] = \
    ground_truth_data[numeric_cols].apply(pd.to_numeric)

bounding_boxes= BoundingBoxes()
#
# Read the output csv file from VIAME and create BoundingBox objects using
# the label with highest confidence for each box.  Then do the same with the
# ground truth csv file
#
with open(DETECTIONS_CSV) as f:
    rows = [line.split(',') for line in f]  # create a list of lists
    rows = rows[2:]
    for row in rows:
        det_id, img_name, frame_id, x1, y1, x2, y2, conf, _ = row[:9]

        det_id = int(det_id)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        conf = float(conf)
        label = "ERR"
        multilabels = row[9:]
        for i in range(0, len(multilabels), 2):  # use label with highest confidence
            label_conf = float(multilabels[i + 1])
            if label_conf == conf:
                label = multilabels[i]
        if "ringed" in label:
            label = "Ringed Seal"
        if "bearded" in label:
            label = "Bearded Seal"
        if "unk" in label:
            label = "UNK Seal"
        if DETECTION_ONLY:
            label = "Seal"
        bbox = BoundingBox(imageName=img_name, classId=label,
                           x=x1, y=y1, w=x2, h=y2, typeCoordinates=CoordinatesType.Absolute,
                           bbType=BBType.Detected, classConfidence=conf, format=BBFormat.XYX2Y2
                           )

        bounding_boxes.addBoundingBox(bbox)
bounding_boxes=bounding_boxes.nms(NMS_THRESH, CONFIDENCE_THRESH) # NMS Step
for index, row in ground_truth_data.iterrows():
    hsId = row["hotspot_id"]
    x1 = row[x1_col]
    x2 = row[x2_col]
    y1 = row[y1_col]
    y2 = row[y2_col]
    label = row['species_id']
    img_name = row['color_image']
    if DETECTION_ONLY:
        label = "Seal"
    bbox = BoundingBox(imageName=img_name, classId=label,
                       x=x1, y=y1, w=x2, h=y2, typeCoordinates=CoordinatesType.Absolute,
                       bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, hsId=hsId
                       )

    bounding_boxes.addBoundingBox(bbox)


evaluator = Evaluator()
metrics = evaluator.GetPascalVOCMetrics(bounding_boxes, IOUThreshold=IOUThreshold,
                                        CONFIDENCE_THRESH=CONFIDENCE_THRESH)
for class_met in metrics:
    label = class_met['class']
    print("%s:"%label)
    tps = class_met["total TP"]
    fps = class_met["total FP"]
    fns = class_met["total FN"]
    if tps ==0 and fps == 0:
        print("No detections for class %s" % label)
        continue
    precision = tps/(tps+fps)
    recall = tps / (tps+fns)
    print("Precision: %f" % precision)
    print("Recall: %f" % recall)
    print("")
# evaluator.PlotPrecisionRecallCurve(bounding_boxes, IOUThreshold=IOUThreshold, showAP=True)
# x=1