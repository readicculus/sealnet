import sys

import cv2
import os

import pandas as pd
import pickle

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

from GroundTruthBoxes import get_ground_truth_boxes

matplotlib.use("TkAgg")
from utils import add_bb_into_image
pkl_file_in = "metrics/pickels/results.pkl"
pkl_all_post_nms = "metrics/pickels/all_post_nms.pkl"
pkl_final_ensemble_boxes = "metrics/pickels/ensemble_bounding_boxes.pkl"
# Can find copies of the pkl files on my google drive
# https://drive.google.com/drive/folders/17_nqRpOdIeAg_iv-T0oAgxSKSp6-R7_d?usp=sharing
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
if os.path.exists(pkl_file_in):
    root_model_dets_dict = pickle.load(open(pkl_file_in, "rb"))
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
    pickle.dump(root_model_dets_dict, open(pkl_file_in, "wb"))


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

ground_truth_boxes = get_ground_truth_boxes()

# for each image we get all bounding boxes and construct model_dets_dict, a dictionary of model_name to detections
# for detection in the current image
plt_im = None
fig = None
if os.path.exists(pkl_final_ensemble_boxes):
    ensembled_boxes = pickle.load(open(pkl_final_ensemble_boxes, "rb"))
else:
    ensembled_boxes = BoundingBoxes()
    total = len(unique_ims)
    for im_idx, im_name in enumerate(unique_ims):
        imbbs = all_together.getBoundingBoxesByImageName(im_name)
        model_dets_dict = {}
        for bb in imbbs:
            if not bb._model in model_dets_dict:
                model_dets_dict[bb._model] = BoundingBoxes()
            model_dets_dict[bb._model].addBoundingBox(bb)

        layers = model_dets_dict.keys()

        # create 3d matrix, one channel per model, increment each pixel per detection box in the corresponding model's channel
        im = Image.open(os.path.join("/data/raw_data/TrainingAnimals_ColorImages/", im_name))
        w,h= im.size
        c = im.layers

        # creating the weight mask which gives pixels a score based on how likely they are to be real detections
        mask = np.zeros((h, w, len(layers)))
        class_mask = np.zeros((h, w, len(layers),2))
        unique_classes = []
        for channel, name in enumerate(layers):
            for box in model_dets_dict[name].getBoundingBoxes():
                x1, y1, x2, y2 = box.getAbsoluteBoundingBox(BBFormat.XYX2Y2)

                if x1 < 0 and x1 > -5:
                    x1 = 0
                elif y1 < 0 and y1 > -5:
                    y1 = 0
                elif x2 > w and x2 < w + 5:
                    x2 = w
                elif y2 > h and y2 < h + 5:
                    y2 = h
                elif x1 < 0 or x2 > w or y1 < 0 or y2 > h:
                    print("err")
                    continue
                confidence = box.getConfidence()
                # confidence^2 allows weight for a detection that was detected by
                # fewer models but with higher confidence, essentially l2 norm across models
                mask[int(y1):int(y2), int(x1):int(x2), channel] += confidence*confidence # L2 norm
                area = class_mask[int(y1):int(y2), int(x1):int(x2), channel]
                id = box.getClassId()
                if id not in unique_classes:
                    unique_classes.append(id)
                class_idx = unique_classes.index(id)+1 # +1 because default (no class) is 0
                area[:,:,1][area[:,:,0]<confidence] = class_idx # must go before we modify confidence
                area[:,:,0][area[:,:,0]<confidence] = confidence

                x=1

        max_idxs = class_mask[:,:,:,0].argmax(axis=2) # maximum confidence for each pixel
        m, n = max_idxs.shape[:2]
        I, J = np.ogrid[:m, :n]
        best_classes = class_mask[I,J,max_idxs,1].astype(np.uint8)
        # binarize
        flat_mask = np.sum(mask, axis=2)  # sum across channels to get flat image
        thresh = np.nanpercentile(flat_mask[flat_mask > 0.0], 90.0)
        flat_mask[flat_mask < thresh] = 0
        flat_mask[flat_mask >= thresh] = 1
        flat_mask = flat_mask.astype(np.uint8)

        # flat_mask_norm = (flat_mask - np.min(flat_mask)) / (np.max(flat_mask) - np.min(flat_mask))

        # find contours in mask then get bounding rectangles
        # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
        contours, hierarchy = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        # create final new boxes for this image and add to ensemble_boxes
        new = 0
        for res_box in bounding_boxes:
            x,y,w,h = res_box
            region = mask[y:y+h, x:x+w,]
            class_region = best_classes[y:y+h, x:x+w,]
            class_idx = np.around(np.mean(class_region), decimals=1).astype(np.int32)-1
            class_label = unique_classes[class_idx]
            r_mean = np.mean(region)     # get mean confidence of all pixels in the region
            confidence = np.sqrt(r_mean) # since confidence is squared we get our new confidence as the sqrt of the mean confidences
            x+= w/2
            y += h/2
            box = BoundingBox(im_name, class_label,x,y,w,h,
                              classConfidence=confidence, bbType=BBType.Ensemble, model="ensemble")
            if box.getArea() < 500:
                continue
            new += 1
            ensembled_boxes.addBoundingBox(box)
        print("%d/%d Image: %s processed, boxes in: %d, boxes out: %d" % (im_idx, total, im_name, len(imbbs), new))
        DRAW = False
        if DRAW:
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
        # free memory, might not need this but sometimes I do not sure how python gc works
        # im = None
        # mask = None
        # flat_mask = None
    pickle.dump(ensembled_boxes, open(pkl_final_ensemble_boxes, "wb"))

x=1