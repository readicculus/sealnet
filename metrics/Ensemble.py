import sys

import cv2
import os

import pandas as pd
import pickle
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
        nms = root_model_dets_dict[model].nms(NMS_THRESH=.5, CONFIDENCE_THRESH=0.1)
        for box in nms._boundingBoxes:
            all_together.addBoundingBox(box)
    pickle.dump(all_together, open(pkl_all_post_nms, "wb"))

# helper function that takes two lists of BoundingBox(object) where bbs1 is a list of detections for an image from one model
# and bbs2 is a list of detections from a different model.  This function returns the matches between the two if the overlap
# is greater than the given iou
# the format of a single match is [bbs1 index, bss2 index, IOU] and this function returns a list of these matches
def model_bbs_cross_correlation(bbs1, bbs2, iou=0):
    evaluator = Evaluator()
    # array of [bbs1 idx, bbs2 idx, iou]
    matches = []
    for i in range(len(bbs1)):
        ious = evaluator._getAllIOUs(bbs1[i], bbs2)
        for ioutup in ious:
            if ioutup[0] > iou:
                matches.append([i, ioutup[3], ioutup[0]])
    return matches

pkl_cross_correlation = "metrics/post_cross_corelation.pkl"
pkl_cross_correlation_nms = "metrics/post_cross_corelation_post_nms.pkl"
good_bbs = BoundingBoxes()
bad_bbs = BoundingBoxes()
if os.path.exists(pkl_cross_correlation) and False:
    (good_bbs,bad_bbs) = pickle.load(open(pkl_cross_correlation, "rb"))
else:
    for im_name in unique_ims:
        imbbs = all_together.getBoundingBoxesByImageName(im_name)
        root_model_dets_dict = {}
        for bb in imbbs:
            if not bb._model in root_model_dets_dict:
                root_model_dets_dict[bb._model] = []
            root_model_dets_dict[bb._model].append(bb)
        mks = list(root_model_dets_dict.keys())
        # match_dict is a dictionary storing the cross correlation matches between models
        # key: model1, value: dictionary{key: model2, value: matches...model3..}
        # the dictionary contains duplicate matches for reverse
        model_match_dict = {}
        for i in range(len(mks)):
            c=mks[i]
            model_match_dict[c]={}
            c_root_dets = root_model_dets_dict[mks[i]] #current model
            others = mks[:i] + mks[i + 1:]
            for o in others:  #to do check if reverse cross correlation already done
                omodel = root_model_dets_dict[o] # other model
                c_matches = model_bbs_cross_correlation(c_root_dets, omodel, iou=.5)
                model_match_dict[c][o] = c_matches

        for c in root_model_dets_dict: # c=current
            c_matches = model_match_dict[c]
            c_root_dets = root_model_dets_dict[c]
            match_ct = [0]*len(c_root_dets)
            for o in c_matches: # o=other (aka one we are comparing to)
                matches = c_matches[o]
                already_matched_idxs = []
                for match in matches:
                    box1_idx = match[0]
                    if box1_idx in already_matched_idxs:
                        # we only want our current bounding box to be matched once
                        # todo we should ideally pic the match with largest iou
                        continue
                    already_matched_idxs.append(match[0])
                    match_ct[box1_idx]+=1
            for mct, box in zip(match_ct, c_root_dets):
                if mct ==3 or \
                        (mct > 1 and box.getConfidence() > .1) \
                        or box.getClassId() == "Polar Bear" \
                        or box.getConfidence() > .3:
                    good_bbs.addBoundingBox(box)
                else:
                    bad_bbs.addBoundingBox(box)
    pickle.dump((good_bbs,bad_bbs), open(pkl_cross_correlation, "wb"))

good_bbs_nms = None
if os.path.exists(pkl_cross_correlation_nms):
    good_bbs_nms = pickle.load(open(pkl_cross_correlation_nms, "rb"))
else:
    good_bbs_nms = good_bbs.nms(.5, 0.0) # already filtered confidence, now we filter nms over all models
    pickle.dump(good_bbs_nms, open(pkl_cross_correlation_nms, "wb"))

plt_im = None
fig = None
# ground_truth_data = pd.read_csv("/home/yuval/Documents/XNOR/sealnet/data/updated_seals.csv", dtype={'hotspot_id': object})

for im_name in unique_ims:
    im = cv2.imread(os.path.join("/data/raw_data/TrainingAnimals_ColorImages/", im_name))
    im=good_bbs_nms.drawAllBoundingBoxes(im, im_name)
    if plt_im is None:
        fig = plt.figure()

        plt_im = plt.imshow(im, cmap='gist_gray_r')
    else:
        plt_im.set_data(im)
        plt_im = plt.imshow(im, cmap='gist_gray_r')
    plt.draw()
    plt.show()
    plt.pause(1)
    plt.cla()
# #
#
# print_models(models)
# nms = {}
# # for model in models:
# #     print_bbs(models[model].nms(.3, .1))
# #
#
# print("ALLLLLLL")
# print_bbs(all_together)
# print("ALLLLLLL NMS")
# atnms = all_together.nms(.3,.3)
# print_bbs(atnms)
#

# x=1