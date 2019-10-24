import os

from PIL import Image

for root, dirs, files in os.walk("/data/raw_data/PolarBears/s3_images/"):
    for f in files:
        ext = f.split(".")[-1]
        if ext == "tif":
            img_path = os.path.join(root, f)
            im = Image.open(img_path)
            out = im.convert("RGB")
            new_name = '.'.join(os.path.basename(img_path).split('.')[:-1]) + ".JPG"
            outfile = os.path.join(root, new_name)
            out.save(outfile, "JPEG", quality=100)
            os.remove(img_path)
