import sys

import cv2
import os

import pandas as pd
import pickle

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

from Evaluator import Evaluator
from BoundingBoxes import BoundingBoxes

matplotlib.use("TkAgg")

from GroundTruthBoxes import get_ground_truth_boxes

pkl_final_ensemble_boxes = "metrics/pickels/ensemble_bounding_boxes.pkl"
df = pd.read_csv("/home/yuval/Documents/XNOR/sealnet/models/darknet/full_detections.csv", delimiter='\t')
unique_ims = set(df["file"].values[0:])

ground_truth_boxes = get_ground_truth_boxes()
ensembled_boxes = pickle.load(open(pkl_final_ensemble_boxes, "rb"))
all_boxes = BoundingBoxes()
all_boxes.addBoundingBoxes(ground_truth_boxes.getBoundingBoxes())
all_boxes.addBoundingBoxes(ensembled_boxes.getBoundingBoxes())
evaluator = Evaluator()

all_boxes = all_boxes.filter_confidence(.3)
res = evaluator.GetPascalVOCMetrics(all_boxes, IOUThreshold=.3)
evaluator.PlotPrecisionRecallCurve(all_boxes, IOUThreshold=.3)

plt_im = None
fig = None
for im_idx, im_name in enumerate(unique_ims):
    img_gts = ground_truth_boxes.getBoundingBoxesByImageName(im_name)
    img_dets = ensembled_boxes.getBoundingBoxesByImageName(im_name)

    im = cv2.imread(os.path.join("/data/raw_data/TrainingAnimals_ColorImages/", im_name))  # load image to get dimensions
    im = ensembled_boxes.drawAllBoundingBoxes(im, im_name)
    im = ground_truth_boxes.drawAllBoundingBoxes(im, im_name)
    # im = all_together.drawAllBoundingBoxes(im, im_name)  # draw raw detections
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)  # draw contours
    if plt_im is None:
        fig = plt.figure()

        plt_im = plt.imshow(im, cmap='gist_gray_r')
    else:
        plt_im.set_data(im)
        plt_im = plt.imshow(im, cmap='gist_gray_r')
    plt.draw()
    plt.show()
    plt.pause(1)
    # plt.waitforbuttonpress(1)
    plt.cla()

    x=1