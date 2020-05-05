import argparse
import glob
import json
import os
import pickle

import requests
from PIL import Image
from hurry.filesize import size

from RemoteDataloader import RemoteChipsDataset


# decided against using database row ids as they are implementation details and may not be consistent if migrated
def im_chip_filename(im_name,x1,y1,x2,y2):
    fn = os.path.splitext(im_name)[0]
    return "%s--%d_%d-%d_%d" % (fn,x1,y1,x2,y2)

w=832
h=832
confidence=.05
path = '/fast/generated_data/fp_chips/%dx%d'%(w,h)
if not os.path.exists(path):
    os.makedirs(path)

CHIPS_ENDPOINT = "http://127.0.0.1:5000/api/false_positives?w=%d&h=%d&confidence=%.2f" %(w,h,confidence)
headers = {'Content-type': 'application/json'}
r = requests.get(url=CHIPS_ENDPOINT)
if r.status_code != 200:
    raise Exception(r.text)
response = json.loads(r.text)

images = response['images']
fp_chips = response['chips']
bytes_saved = 0
i=0
total = len(fp_chips)
for x in fp_chips:
    im = images[x]
    chips = fp_chips[x]
    chip_im = Image.open(im['file_path'])
    for chip in chips:
        area = (chip['x1'], chip['y1'], chip['x2'], chip['y2'])
        chip_im = chip_im.crop(area)
        fn = im_chip_filename(im['file_name'], chip['x1'], chip['y1'], chip['x2'], chip['y2'])
        chip_fp = os.path.join(path, "%s.JPG" % fn)
        chip_im.save(chip_fp, quality=90)
        bytes_saved += os.stat(chip_fp).st_size

        # save empty text file for YOLO label
        with open(os.path.join(path, "%s.txt" % fn), 'w') as fp:
            pass

    if i % 50 == 0:
        print("%d/%d images processed" % (i, total))
        print("byte_saved:%s" % size(bytes_saved))

    i+=1

for jpgfile in glob.glob(os.path.join(path, "*.JPG")):
    filename, file_extension = os.path.splitext(jpgfile)
    txt_filename = filename+".txt"
    with open(txt_filename, 'w') as fp:
        pass


