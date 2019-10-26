from Evaluator import Evaluator
from BoundingBox import *
from utils import *
import numpy as np


class BoundingBoxes:
    def __init__(self):
        self._boundingBoxes = []
        self._images = {}

    def addBoundingBox(self, bb):
        self._boundingBoxes.append(bb)
        if not bb.getImageName() in self._images:
            self._images[bb.getImageName()] = []
        self._images[bb.getImageName()].append(bb)
    def addBoundingBoxes(self, bbs):
        for bb in bbs:
            self.addBoundingBox(bb)
    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d, _boundingBox):
                del self._boundingBoxes[d]
                break
        for d in self._images[_boundingBox.bb.getImageName()]:
            if BoundingBox.compare(d, _boundingBox):
                del self._images[_boundingBox.bb.getImageName()][d]
                break
        return

    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        if imageName in self._images:
            return self._images[imageName]
        return []

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        box_label = "%s %.4f" % (self.getClassId(), self._classConfidence)

        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0), label=box_label)  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0), label=box_label)  # red
        return image

    def nms(self, NMS_THRESH, CONFIDENCE_THRESH):
        if NMS_THRESH == 0:
            return self
        evaluator = Evaluator()
        images = set()
        newBoundingBoxes = BoundingBoxes()

        for bb in self._boundingBoxes:
            images.add(bb.getImageName())
        for img_idx,image in enumerate(images):
            bboxes = self.getBoundingBoxesByImageName(image)
            gts = [bb for bb in bboxes if bb.getBBType() == BBType.GroundTruth]
            dets = [bb for bb in bboxes if bb.getBBType() == BBType.Detected and bb.getConfidence() >= CONFIDENCE_THRESH]
            duplicates = np.zeros((len(dets), len(dets)))
            for i, det in enumerate(dets):
                det_abs = det.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                for i2, det2 in enumerate(dets):
                    if i == i2 or duplicates[i2][i] >0 or duplicates[i][i2] >0:
                        continue
                    det2_abs = det2.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                    a=evaluator.iou(det_abs, det2_abs)
                    if a > NMS_THRESH:
                        duplicates[i2][i] = 2  # set to 2 if > NMS thresh
                    else:
                        duplicates[i2][i] = 1 # set to 1 if already see so we don't check in reverse order

            good_idxs = np.ones(len(dets))
            if np.sum(duplicates) > 0:
                nms_to_filter=np.argwhere(duplicates==2)
                for idxs in nms_to_filter:
                    if dets[idxs[0]].getConfidence() > dets[idxs[1]].getConfidence():
                        good_idxs[1] = 0
                    else:
                        good_idxs[0] = 0


            good_idxs = good_idxs.astype(np.bool)
            for i in range(len(dets)):
                if good_idxs[i]:
                    newBoundingBoxes.addBoundingBox(dets[i])

            for box in gts:
                newBoundingBoxes.addBoundingBox(box)
        return newBoundingBoxes

    def filter_confidence(self, conf):
        new = BoundingBoxes()
        for bb in self._boundingBoxes:
            if bb.getConfidence() >= conf or bb.getBBType() == BBType.GroundTruth:
                new.addBoundingBox(bb)
        return new
    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image