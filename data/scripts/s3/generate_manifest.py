import os

import boto3
import boto
import json
import pandas as pd
import xml.etree.ElementTree as ET
import math
from PIL import Image

BUCKET_URI = "https://arcticseals.s3-us-west-2.amazonaws.com"
BUCKET_URI = "s3://arcticseals/"

annotations_path_chess = "/data/raw_data/PolarBears/2016CHESS_PolarBearAnnotations.csv"
annotations_path_2019 = "/data/raw_data/PolarBears/2019_pb_annotations.csv"
numeric_cols = ["Xmin", "Ymin", "Xmax", "Ymax"]


data2016 = pd.read_csv(annotations_path_chess, sep = ',', header=0, dtype={'PB_ID': object})
data2019 = pd.read_csv(annotations_path_2019, sep = ',', header=0, dtype={'PB_ID': object})
data2016[numeric_cols] =  data2016[numeric_cols].apply(pd.to_numeric)
data2019[numeric_cols] =  data2019[numeric_cols].apply(pd.to_numeric)
data219_images = data2019.copy()
for idx, row in data2019.iterrows():
    status_str = ""
    if row["Poor_image_quality"] == "Y":
        status_str += "bad_res,"
    if str(row["Age_class"]) != "nan":
        status_str += row["Age_class"]
    xml_file = os.path.join("/data/raw_data/PolarBears/Polar Bear Imagery/2019_Beaufort_PolarBears",
                            row["Frame_xml"])
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img = ".".join(root[1].text.split('.')[:-1])+".JPG"
    data219_images.set_value(idx, 'Frame_xml', img)
data2019 = data219_images

###### S3 stuff
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('arcticseals')
conn = boto.connect_s3()
bucket = conn.get_bucket("arcticseals")

data_list = []

for file in my_bucket.objects.filter(Prefix='polarbears'):
    print(file)
    if file.key.split(".")[-1] != "JPG":
        continue
    if "2019_Beaufort_PolarBears" in file.key and not "rgb" in file.key:
        continue
    if "THERM-8-BIT"  in file.key:
        continue
    KEY = bucket.get_key(file.key)
    resource_uri = BUCKET_URI + file.key
    h,w,c = KEY.get_metadata("height"), KEY.get_metadata("width"), KEY.get_metadata("depth")

    file_name = os.path.basename(file.key)
    data1 = data2019.loc[data2019['Frame_xml'] == file_name]
    data2 = data2016.loc[data2016['Frame_color'] == file_name]
    xmins = list(data1["Xmin"])+list(data2["Xmin"])
    xmaxs = list(data1["Xmax"])+list(data2["Xmax"])
    ymins = list(data1["Ymin"])+list(data2["Ymin"])
    ymaxs = list(data1["Ymax"])+list(data2["Ymax"])
    ids = list(data1["PB_ID"])+list(data2["PB_ID"])

    label_attr_name = "NOAAlabelsdone"
    NUM_OF_BOXES = len(ids)
    data = {}
    data["source-ref"] = resource_uri
    if NUM_OF_BOXES == 0:
        label_attr_name = "NOAAlabelstodo"
    data[label_attr_name] = {
        "annotations": [],
        "image_size": [{
            "width": w,
            "height": h,
            "depth": c
        }]}
    if NUM_OF_BOXES > 0:
        for i in range(len(ids)):
            xmin, xmax, ymin, ymax = xmins[i], xmaxs[i], ymins[i], ymaxs[i]
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w/2
            cy = ymin + h/2
            if math.isnan(cy) or math.isnan(cx) or math.isnan(w) or math.isnan(h):
                continue
            annotation = {
              "class_id": 0,
              "width": w,
              "top": cy,
              "height": h,
              "left": cx
            }
            data[label_attr_name]["annotations"].append(annotation)

        job_name = "labeling-job/polarbears"
        data[label_attr_name+"-metadata"] = {
            "job-name": job_name,
            "class-map": {
                "0": "Polar Bear"
            },
            "human-annotated": "yes",
            "objects": [{
                "confidence": 1
            }],
            "creation-date": "2019-08-01T04:25:44.859597",
            "type": "groundtruth/object-detection"
        }
    data_list.append(data)
for data in data_list:
    with open("polarbears.manifest", "a") as f:
        json.dump(data, f, sort_keys=True)
        f.write("\r\n")

x=1
