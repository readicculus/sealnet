import pandas as pd

from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from utils import *

df = pd.read_csv("/home/yuval/Documents/XNOR/sealnet/models/darknet/test.csv", delimiter='\t')


def print_models(models):
    print("Total detections:")
    total = 0
    for model in models:
        print(model)
        bbs = models[model]
        ct = len(bbs._boundingBoxes)
        total += ct
        class_ct = {}
        for bb in bbs._boundingBoxes:
            if bb.getClassId() not in class_ct:
                class_ct[bb.getClassId()] = 0

            class_ct[bb.getClassId()] += 1
        for c in class_ct:
            print("%s %d" % (c, class_ct[c]))
        print("%d total" % ct)
        print("")

    print("%d total" % total)

models = {}
nms = {}
for i, det in df.iloc[1:].iterrows():
    if i > 10000:
        break
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
                      classConfidence=confidence, bbType=BBType.Detected)
    models[model].addBoundingBox(box)

print_models(models)
for model in models:
    nms[model] = models[model].nms(.3, .1)
print_models(nms)

x=1