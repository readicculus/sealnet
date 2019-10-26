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
models = {}

if os.path.exists(pkl_file):
    # models = pickle.load(open(pkl_file, "rb"))
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
        if not model in models:
            models[model] = BoundingBoxes()

        box = BoundingBox(imageName, label,x,y,w,h,
                          classConfidence=confidence, bbType=BBType.Detected, model=model)
        models[model].addBoundingBox(box)
    pickle.dump( models, open( pkl_file, "wb" ) )

## PUT ALL BBOXES IN ONE LIST BUT FIRST NMS OVER INDAVIDUAL MODELS

if os.path.exists(pkl_all_post_nms):
    all_together = pickle.load(open(pkl_all_post_nms, "rb"))
else:
    all_together = BoundingBoxes()
    for model in models:
        nms = models[model].nms(NMS_THRESH=.5, CONFIDENCE_THRESH=0.05)
        for box in nms._boundingBoxes:
            all_together.addBoundingBox(box)
    pickle.dump(all_together, open(pkl_all_post_nms, "wb"))

def bbs_ious(bbs1, bbs2, iou=0):
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
if os.path.exists(pkl_cross_correlation):
    (good_bbs, bad_bbs) = pickle.load(open(pkl_all_post_nms, "rb"))
else:
    for im_name in unique_ims:
        imbbs = all_together.getBoundingBoxesByImageName(im_name)
        models = {}
        for bb in imbbs:
            if not bb._model in models:
                models[bb._model] = []
            models[bb._model].append(bb)
        mks = list(models.keys())
        match_dict = {}
        for i in range(len(mks)):
            c=mks[i]
            match_dict[c]={}
            cmodel = models[mks[i]] #current model
            others = mks[:i] + mks[i + 1:]
            for o in others:
                # if o in match_dict and c in match_dict[o]:
                #     continue
                omodel = models[o] # other model
                matches = bbs_ious(cmodel, omodel,iou=.3)
                match_dict[c][o] = matches

        for c in models:
            matches = match_dict[c]
            cmodel = models[c]
            match_ct = [0]*len(cmodel)
            for o in matches:
                omatches = matches[o]
                for match in omatches:
                    match_ct[match[0]]+=1
            for mct, box in zip(match_ct, cmodel):
                if mct > 0 or box.getClassId() == "Polar Bear" or box.getConfidence() > .3:
                    good_bbs.addBoundingBox(box)
                else:
                    bad_bbs.addBoundingBox(box)
    pickle.dump(good_bbs, open(pkl_cross_correlation, "wb"))

good_bbs_nms = None
if os.path.exists(pkl_cross_correlation_nms):
    good_bbs_nms = pickle.load(open(pkl_cross_correlation_nms, "rb"))
else:
    good_bbs_nms = good_bbs.nms(.5, 0.0) # already filtered confidence, now we filter nms over all models!!@
    pickle.dump(good_bbs_nms, open(pkl_cross_correlation_nms, "wb"))


for im_name in unique_ims:
    plt_im = None
    fig = None
    im = cv2.imread(os.path.join("/data/raw_data/TrainingAnimals_ColorImages/", im_name))
    good_bbs_nms.drawAllBoundingBoxes(im, im_name)

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
#
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