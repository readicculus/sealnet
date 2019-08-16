import pandas as pd
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

annotations_path_chess = "/data/raw_data/PolarBears/2016CHESS_PolarBearAnnotations.csv"
annotations_path_2019 = "/data/raw_data/PolarBears/2019_pb_annotations.csv"
numeric_cols = ["Xmin", "Ymin", "Xmax", "Ymax"]

base_img_dir = "/data/raw_data/PolarBears/"
out_dir = "/fast/raw_data/PolarBears/"

data2016 = pd.read_csv(annotations_path_chess, sep = ',', header=0, dtype={'PB_ID': object})
data2019 = pd.read_csv(annotations_path_2019, sep = ',', header=0, dtype={'PB_ID': object})
data2016[numeric_cols] =  data2016[numeric_cols].apply(pd.to_numeric)
data2019[numeric_cols] =  data2019[numeric_cols].apply(pd.to_numeric)

# merged = pd.DataFrame(columns=['color_image','thermal_image',
#                                'hotspot_id','hotspot_type',
#                                'species_id','species_confidence',
#                                'fog','thermal_x','thermal_y',
#                                'color_left','color_top','color_right',
#                                'color_bottom','updated_left','updated_top',
#                                'updated_right','updated_bottom',
#                                'updated','status'])

new_rows = []
for idx, row in data2016.iterrows():
    status_str = ""
    if row["Poor_image_quality"] == "Y":
        status_str += "bad_res,"
    if str(row["Age_class"]) != "nan":
        status_str += row["Age_class"]
    new_row = {
     'color_image':row["Frame_color"],
     'thermal_image':"",
     'hotspot_id':row["PB_ID"],
     'hotspot_type':"Animal",
     'species_id':"Polar Bear",
     'species_confidence':"",
     'fog':"",
     'thermal_x':"",
     'thermal_y':"",
     'color_left':row["Xmin"],
     'color_top':row["Ymin"],
     'color_right':row["Xmax"],
     'color_bottom':row["Ymax"],
     'updated_left':row["Xmin"],
     'updated_top':row["Ymin"],
     'updated_right':row["Xmax"],
     'updated_bottom':row["Ymax"],
     'updated':True,
     'status':status_str}
    new_rows.append(new_row)
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
    img = root[1].text
    new_row = {
     'color_image':img,
     'thermal_image':"",
     'hotspot_id':row["PB_ID"],
     'hotspot_type':"Animal",
     'species_id':"Polar Bear",
     'species_confidence':"",
     'fog':"",
     'thermal_x':"",
     'thermal_y':"",
     'color_left':row["Xmin"],
     'color_top':row["Ymin"],
     'color_right':row["Xmax"],
     'color_bottom':row["Ymax"],
     'updated_left':row["Xmin"],
     'updated_top':row["Ymin"],
     'updated_right':row["Xmax"],
     'updated_bottom':row["Ymax"],
     'updated':True,
     'status':status_str}
    new_rows.append(new_row)
merged = pd.DataFrame(new_rows)

# filter out nans
merged = merged[merged.color_image.notnull()]
merged = merged[merged.color_right.notnull()]


images = list(merged['color_image'])
found_images = []
found_images_full_path = []
for root, dirs, files in os.walk(base_img_dir):
    for f in files:
        if f in images:
            img_path = os.path.join(root, f)
            found_images_full_path.append(img_path)
            found_images.append(f)
notfound = [img for img in images if img not in found_images]
merged = merged[merged.color_image.isin(found_images)]
merged = merged.astype({'color_image': np.str,'thermal_image': np.str,
                               'hotspot_id': np.str,'hotspot_type': np.str,
                               'species_id': np.str,'species_confidence': np.str,
                               'fog': np.str,'thermal_x': np.str,'thermal_y': np.str,
                               'color_left': np.int,'color_top': np.int,
                               'color_right': np.int,
                               'color_bottom': np.int,'updated_left': np.int,
                               'updated_top': np.int,
                               'updated_right': np.int,'updated_bottom': np.int,
                               'updated': np.bool,'status': np.str})
for image_path in found_images_full_path:
    im = Image.open(image_path)
    out = im.convert("RGB")
    new_name = '.'.join(os.path.basename(image_path).split('.')[:-1])+".jpg"
    outfile = os.path.join(out_dir, new_name)
    out.save(outfile, "JPEG", quality=100)
out_csv = os.path.join(out_dir, new_name)

