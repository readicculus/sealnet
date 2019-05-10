from operator import itemgetter
from random import randint
import numpy as np
from imgaug import BoundingBox, BoundingBoxesOnImage
def random_shift(max):
    dx = randint(-max, max)
    dy = randint(-max, max)
    return dx, dy

def bbox_is_in(parent, child):
    return parent.contains((child.x1, child.y1)) and parent.contains((child.x2, child.y2))
def crop_around_hotspots(bbs, crop_size):
    h,w,c = bbs.shape

    chips = []
    for bbox_idx, bbox in enumerate(bbs.bounding_boxes):
        center_x = bbox.center_x
        center_y = bbox.center_y
        half = np.array(crop_size)/2

        crop_box = BoundingBox(x1 = center_x-half[0], y1=center_y-half[0],x2 = center_x+half[0], y2=center_y+half[0])
        count = 0
        while count < 10000:
            dx, dy = random_shift(crop_size[0]/2)
            crop_box = crop_box.shift(left=dx, top=dy)
            if crop_box.is_fully_within_image(bbs.shape) and bbox_is_in(crop_box, bbox):
                break

            crop_box = BoundingBox(x1 = center_x-half[0], y1=center_y-half[0],x2 = center_x+half[0], y2=center_y+half[0])
            count+=1
        if count != 0:
            print(count)

        others_in_same_chip_idx = []
        others_in_same_chip = []
        for bbox_idx2, bbox2 in enumerate(bbs.bounding_boxes):
            if bbox_idx2 == bbox_idx:
                continue

            if bbox_is_in(crop_box, bbox2):
                others_in_same_chip_idx.append(bbox_idx2)
                others_in_same_chip.append(bbs.bounding_boxes[bbox_idx2])


        hotspots = [bbox] + others_in_same_chip
        print("hs %d" % len(hotspots))
        chips.append((crop_box, hotspots))
    print("chips %d" % len(chips))

    return chips


# sliding window over image return tuple of cropped regions
# and location in original image
def full_image_tile_crops(image, width=640, height=640):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)

    stride_rows = int(height - (height-_m)/nrows)
    stride_cols = int(width - (width-_n)/ncols)

    chips = []
    for i in range(ncols+1):
        for j in range(nrows+1):
            dims = (j*stride_rows,j*stride_rows+height,i*stride_cols,i*stride_cols+width)
            cropped_img = image[j*stride_rows:j*stride_rows+height,i*stride_cols:i*stride_cols+width]
            if cropped_img.shape[0] != height or cropped_img.shape[1] != width:
                print("Sizes wrong")
                print(cropped_img.shape)
            chips.append((cropped_img, dims))
    return chips