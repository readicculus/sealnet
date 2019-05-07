from random import randint


def get_tile_images(image, width=640, height=640):
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
            cropped_img = image[j*stride_rows:j*stride_rows+height,i*stride_cols:i*stride_cols+width]
            if cropped_img.shape[0] != height or cropped_img.shape[1] != width:
                print("Sizes wrong")
                print(cropped_img.shape)
            chips.append(cropped_img)


def recalculate_crops(rgb_bb_b, rgb_bb_t, rgb_bb_l, rgb_bb_r, imgh, imgw, maxShift, minShift, crop_size):
    # center points of bounding box in the image
    center_y_global = rgb_bb_t + (rgb_bb_b - rgb_bb_t) / 2
    center_x_global = rgb_bb_l + (rgb_bb_r - rgb_bb_l) / 2

    lcrop_orig = center_x_global - crop_size/2
    rcrop_orig = center_x_global + crop_size/2
    tcrop_orig = center_y_global - crop_size/2
    bcrop_orig = center_y_global + crop_size/2



    dx, dy = random_shift(tcrop_orig, bcrop_orig, lcrop_orig, rcrop_orig, imgw, imgh, minShift, maxShift)

    lcrop = lcrop_orig + dx
    rcrop = rcrop_orig + dx
    bcrop = bcrop_orig + dy
    tcrop = tcrop_orig + dy

    # Ensure hotspot is still in cropped space, if not shift so that it is
    if center_x_global < lcrop:
        diff = lcrop - center_x_global
        lcrop -= diff
        rcrop -= diff

    if center_x_global > rcrop:
        diff = center_x_global - rcrop
        lcrop += diff
        rcrop += diff

    if center_y_global < tcrop:
        diff = center_y_global - tcrop
        bcrop += diff
        tcrop += diff

    if center_y_global > bcrop:
        diff = bcrop - center_y_global
        bcrop -= diff
        tcrop -= diff

    if tcrop < 0:
        diff = 0 - tcrop
        tcrop += diff
        bcrop += diff
    if bcrop > imgh:
        diff = bcrop - imgh
        bcrop -= diff
        tcrop -= diff
    if lcrop < 0:
        diff = 0 - lcrop
        lcrop += diff
        rcrop += diff
    if rcrop > imgw:
        diff = rcrop - imgw
        rcrop -= diff
        lcrop -= diff



    dx = lcrop_orig - lcrop
    dy = tcrop_orig - tcrop
    local_x = crop_size/2 + dx
    local_y = crop_size/2 + dy
    return tcrop, bcrop, lcrop, rcrop, local_x, local_y, dx, dy


def random_shift(topCrop, bottomCrop, leftCrop, rightCrop, w, h, minShift, maxShift):
    if maxShift == 0:
        return 0, 0

    dx = 0
    dy = 0

    # make dx
    if leftCrop != 0 and randint(0, 1) == 1:
        dx -= randint(minShift, maxShift)

    elif rightCrop != 0 and randint(0, 1) == 1:
        dx += randint(minShift, maxShift)

    # left crop outside image bounds
    if not leftCrop + dx > 0:
        dx = 0
    if not rightCrop + dx < w:
        dx = 0

    # make dy
    if topCrop != 0 and randint(0, 1) == 1:
        dy -= randint(minShift, maxShift)

    elif bottomCrop != 0 and randint(0, 1) == 1:
        dy += randint(minShift, maxShift)

    # left crop outside iomage bounds
    if topCrop + dy < 0:
        dy = 0
    if bottomCrop + dx > h:
        dy = 0

    return dx, dy