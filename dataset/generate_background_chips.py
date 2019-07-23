#inputs
# folder of background images
# size to chip
# output folder

#outputs
# chips to folder
import os
import argparse
import cv2
import math
from PIL import Image
import glob
import random
from hurry.filesize import size

parser = argparse.ArgumentParser(description='Generate background chips.')
parser.add_argument('-in', '--input_path', dest='input_path', required=True)
parser.add_argument('-out', '--output_path', dest='output_path', required=True)
parser.add_argument('-n','--num_images',
                    default=100, help='Number of images to process, '
                                      'program randomly selects n from all available images', type=int)
parser.add_argument('--height',
                    default=416, help='Chip height', type=int)
parser.add_argument('--width',
                    default=416, help='Chip width', type=int)

parser.add_argument('--cpi',
                    default=8, help='Number of chips to crop from each image', type=int)

args = parser.parse_args()

if not os.path.isdir(args.input_path):
    print("Input folder does not exist")
    quit()
if not os.path.isdir(args.output_path):
    print("Output folder does not exist")
    quit()

filename_list = glob.glob(os.path.join(args.input_path, '*.JPG'))
if args.num_images < len(filename_list):
    filename_list = random.sample(filename_list, args.num_images)
else:
    args.num_images = len(filename_list)
num_chips_created = 0
total_bytes = 0
for idx, filename in enumerate(filename_list):
    print("Processed %d/%d - %d chips created - total bytes written %s" % (idx, args.num_images, num_chips_created,
                                                                           size(total_bytes)))
    filename_base = os.path.join(args.output_path, os.path.basename(filename).split(".")[0])

    im = cv2.imread(filename)
    width = im.shape[1] - im.shape[1] % args.width
    height = im.shape[0] - im.shape[0] % args.height
    path = os.path.basename(filename)
    chips_in_im = 0
    total_chips_in_im = int(width/args.width) * int(height/args.height)
    skip = int(total_chips_in_im / args.cpi)
    for i in range(int(width/args.width)):
        for j in range(int(height/args.height)):
            chips_in_im += 1
            if chips_in_im % skip == 0:
                crop_img = im[args.height * j:args.height * (j+1),args.width * i:args.width * (i+1)]
                out_filename = filename_base + "-%d-%d" % (i,j)
                cv2.imwrite(out_filename + ".jpg", crop_img)
                total_bytes += os.path.getsize(out_filename + ".jpg")
                with open(out_filename + ".txt", 'w'): pass
                num_chips_created += 1
