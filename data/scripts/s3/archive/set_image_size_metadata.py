import os

import boto
import boto3
import boto.s3.connection
import json
import pandas as pd
import xml.etree.ElementTree as ET

from PIL import Image
import os, fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


conn = boto.connect_s3()
bucket = conn.get_bucket("arcticseals")

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('arcticseals')

file_list = []
label_attr_name = "NOAA-labeled"


for file in my_bucket.objects.filter(Prefix='polarbears'):
    print(file)
    if file.key.split(".")[-1] != "JPG":
        continue
    if "2019_Beaufort_PolarBears" in file.key and not "rgb" in file.key:
        continue
    result = find(os.path.basename(file.key), '/data/raw_data/PolarBears/s3_images')
    if len(result) == 0:
        print("could not find file for %s" % file.key)
    local_file = result[0]
    im = Image.open(local_file)

    key = bucket.get_key(file.key)
    key.set_metadata('width', im.width)
    key.set_metadata('height', im.height)
    key.set_metadata('depth', im.layers)
    key.set_remote_metadata({'width': im.width,
                             'height': im.height,
                             'depth': im.layers}, {}, True)

