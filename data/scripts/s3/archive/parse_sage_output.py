import pandas as pd

import boto3
import sagemaker
import os
import json
LOCAL_S3 = "/data/raw_data/PolarBears/s3_images/"
sess = sagemaker.Session()
s3 = boto3.resource('s3')

training_image = sagemaker.amazon.amazon_estimator.get_image_uri(
    boto3.Session().region_name, 'object-detection', repo_version='latest')

augmented_manifest_filename_train = 'output.manifest' # Replace with the filename for your training data.
augmented_manifest_filename_validation = 'output.manifest' # Replace with the filename for your training data.
bucket_name = "arcticseals" # Replace with your bucket name.
s3_prefix = 'polarbears/labels/FFFF/manifests/output' # Replace with the S3 prefix where your data files reside.

s3_train_data_path = 's3://{}/{}/{}'.format(bucket_name, s3_prefix, augmented_manifest_filename_train)

augmented_manifest_s3_key = s3_train_data_path.split(bucket_name)[1][1:]
s3_obj = s3.Object(bucket_name, augmented_manifest_s3_key)
augmented_manifest = s3_obj.get()['Body'].read().decode('utf-8')
augmented_manifest_lines = augmented_manifest.split('\n')

num_training_samples = len(augmented_manifest_lines) # Compute number of training samples for use in training job request.

# annotations_path_chess = "/data/raw_data/PolarBears/2016CHESS_PolarBearAnnotations.csv"
# annotations_path_2019 = "/data/raw_data/PolarBears/2019_pb_annotations.csv"
# numeric_cols = ["Xmin", "Ymin", "Xmax", "Ymax"]
#
# base_img_dir = "/data/raw_data/PolarBears/"
# out_dir = "/fast/raw_data/PolarBears/"
#
# data2016 = pd.read_csv(annotations_path_chess, sep = ',', header=0, dtype={'PB_ID': object})
# data2019 = pd.read_csv(annotations_path_2019, sep = ',', header=0, dtype={'PB_ID': object})
# data2016[numeric_cols] =  data2016[numeric_cols].apply(pd.to_numeric)
# data2019[numeric_cols] =  data2019[numeric_cols].apply(pd.to_numeric)
# data2016 = data2016[data2016["Frame_color"].notnull()]
# data2019 = data2019[data2019["Frame_xml"].notnull()]
new_rows = []

for i in range(len(augmented_manifest_lines)):
    el = augmented_manifest_lines[i]
    if el == "":
        continue
    obj = json.loads(el)
    src = obj["source-ref"]
    path = "/".join(src.replace("s3://","").split("/")[2:])
    img_local = os.path.join(LOCAL_S3, path)
    filename = ".".join(os.path.basename(img_local).split(".")[:-1])

    # row = data2016[data2016["Frame_color"].str.contains(filename)]
    # row2 = data2019[data2019["Frame_xml"].str.contains(filename)]
    human_labeled = obj['NOAAlabelsdone']
    labels = human_labeled['annotations']
    for label in labels:
        cx,cy,w,h = label['left'],label['top'], label['width'], label['height']
        x1 = int(cx - (w/2))
        x2 = int(cx + (w/2))
        y1 = int(cy - (h/2))
        y2 = int(cy + (h/2))
        new_row = {
            'color_image': os.path.basename(img_local),
            'thermal_image': "",
            'hotspot_id': "TBD",
            'hotspot_type': "Animal",
            'species_id': "Polar Bear",
            'species_confidence': "",
            'fog': "",
            'thermal_x': "",
            'thermal_y': "",
            'color_left': x1,
            'color_top': y1,
            'color_right': x2,
            'color_bottom': y2,
            'updated_left': x1,
            'updated_top': y1,
            'updated_right': x2,
            'updated_bottom': y2,
            'updated': True,
            'status': 'TBD'}
        new_rows.append(new_row)
    print(os.path.isfile(img_local))
new_df = pd.DataFrame(new_rows)
new_df.to_csv("polar_bears.csv")
