import sys

import cv2
import os

import pandas as pd
import pickle

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from utils import add_bb_into_image
pkl_all_post_nms = "metrics/all_post_nms.pkl"
pkl_file = "metrics/results.pkl"
from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from utils import *
from Evaluator import Evaluator
def print_bbs(bbs):
    ct = len(bbs._boundingBoxes)
    class_ct = {}
    for bb in bbs._boundingBoxes:
        if bb.getClassId() not in class_ct:
            class_ct[bb.getClassId()] = 0

        class_ct[bb.getClassId()] += 1
    for c in class_ct:
        print("%s %d" % (c, class_ct[c]))
    print("%d total" % ct)
    print("")

def print_models(models):
    print("Total detections:")
    for model in models:
        print(model)
        bbs = models[model]
        print_bbs(bbs)

df = pd.read_csv("/home/yuval/Documents/XNOR/sealnet/models/darknet/full_detections.csv", delimiter='\t')
unique_ims = set(df["file"].values[0:])
root_model_dets_dict = {}

# first convert inference output to pkl file that is a ditionary key:model_name value:BoundingBoxes(object)
if os.path.exists(pkl_file):
    root_model_dets_dict = pickle.load(open(pkl_file, "rb"))
    pass
else:
    for i, det in df.iloc[1:].iterrows():
        imageName=det["file"]
        model=det["weights"]
        label=det["label"]
        confidence=det["confidence"]
        x=det["x"]
        y=det["y"]
        w=det["w"]
        h=det["h"]
        if not model in root_model_dets_dict:
            root_model_dets_dict[model] = BoundingBoxes()

        box = BoundingBox(imageName, label,x,y,w,h,
                          classConfidence=confidence, bbType=BBType.Detected, model=model)
        root_model_dets_dict[model].addBoundingBox(box)
    pickle.dump(root_model_dets_dict, open(pkl_file, "wb"))


# for each model we use nms to remove redundant labels detected by that model
# we then aggregate all detections across all models into one BoundingBoxes(object)
if os.path.exists(pkl_all_post_nms):
    all_together = pickle.load(open(pkl_all_post_nms, "rb"))
else:
    all_together = BoundingBoxes()
    for model in root_model_dets_dict:
        nms = root_model_dets_dict[model].multi_class_nms(NMS_THRESH=.5, CONFIDENCE_THRESH=0.1)
        for box in nms._boundingBoxes:
            all_together.addBoundingBox(box)
    pickle.dump(all_together, open(pkl_all_post_nms, "wb"))

# for each image we get all bounding boxes and construct model_dets_dict, a dictionary of model_name to detections
# for detection in the current image
plt_im = None
fig = None
for im_name in unique_ims:
    imbbs = all_together.getBoundingBoxesByImageName(im_name)
    model_dets_dict = {}
    for bb in imbbs:
        if not bb._model in model_dets_dict:
            model_dets_dict[bb._model] = BoundingBoxes()
        model_dets_dict[bb._model].addBoundingBox(bb)

    layers = model_dets_dict.keys()

    # create 3d matrix, one channel per model, increment each pixel per detection box in the corresponding model's channel
    im = cv2.imread(os.path.join("/data/raw_data/TrainingAnimals_ColorImages/", im_name))  # load image to get dimensions
    h,w,c = im.shape

    # creating the weight mask which gives pixels a score based on how likely they are to be real detections
    mask = np.zeros((h, w, len(layers)))
    for channel, name in enumerate(layers):
        for box in model_dets_dict[name].getBoundingBoxes():
            x1, y1, x2, y2 = box.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            confidence = box.getConfidence()
            # confidence^2 allows weight for a detection that was detected by
            # fewer models but with higher confidence
            mask[int(y1):int(y2), int(x1):int(x2), channel] += confidence*confidence

    # binarize
    flat_mask = np.sum(mask, axis=2)  # sum across channels to get flat image
    thresh = np.percentile(flat_mask[flat_mask > 0], 90)
    flat_mask[flat_mask < thresh] = 0
    flat_mask[flat_mask >= thresh] = 1
    flat_mask = flat_mask.astype(np.uint8)

    # flat_mask_norm = (flat_mask - np.min(flat_mask)) / (np.max(flat_mask) - np.min(flat_mask))

    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    ensembled_boxes = BoundingBoxes()
    for res_box in bounding_boxes:
        x,y,w,h = res_box
        region = mask[y:y+h, x:x+w,]
        r_mean = np.mean(region)     # get mean confidence of final region
        confidence = np.sqrt(r_mean) # since mask was confidence^2
        x+= w/2
        y += h/2
        box = BoundingBox(im_name, "detection",x,y,w,h,
                          classConfidence=confidence, bbType=BBType.Ensemble, model="ensemble")
        if box.getArea() < 500:
            continue
        ensembled_boxes.addBoundingBox(box)

    im = ensembled_boxes.drawAllBoundingBoxes(im, im_name)
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
    # plt.pause(1)
    plt.waitforbuttonpress(1)
    plt.cla()
    # a = Image.fromarray(flat_mask*255)
    # a.save('out.jpg')
    x=1