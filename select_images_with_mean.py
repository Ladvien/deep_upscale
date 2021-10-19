"""
When I was mixing icons and magic symbols for training data,
I noticed the GAN struggling.  It appeared as it was trying to
decide if the image should be mostly black or mostly white. This 
seemed to always lead to a mode collapse and devolution into noise.

To combat this, I sorted the icon images into those which are lighter
than the mean of the magic symbols subtracting a threshold.

It would be better to look at distributions and standard deviations
from the magic symbol means, but this was fast and seemed to do the 
trick.
"""
import os
import sys
import glob

import numpy as np
import cairosvg
from PIL import Image
from random import randint
import cv2

from libs.img_mods import add_noise, invert_mostly_black_images
from libs.img_utils import ImageUtils

import matplotlib.pyplot as plt

img_utils = ImageUtils()

#############
# Setup
#############
root_path = os.environ["HOME"]
project_path = f"{root_path}/deep_upscale/"


#################
# Parameters
#################
input_path = f"{root_path}/deep_upscale/images/raw/"
output_path = f"{root_path}/deep_upscale/images/"

poor_images_path = f"{output_path}poor/"
rich_images_path = f"{output_path}rich/"

magic_image_path = "/home/ladvien/Documents/magic_symbols/"

print(rich_images_path)

##########################
# Sort Images on Darkness
##########################
png_file_paths = glob.glob(f"{rich_images_path}/*.png")
magic_file_paths = glob.glob(f"{magic_image_path}/*.png")


magic_means = [
    np.array(Image.open(img_path)).mean() for img_path in magic_file_paths
]

magic_img_mean = np.array(magic_means).mean()
print(f"Mean of magic symbol images: {magic_img_mean}")

other_means = [
    np.array(Image.open(img_path)).mean() for img_path in png_file_paths
]

print(f"Mean of icons: {np.array(other_means).mean()}")


sorted_dir = f"{output_path}sorted_by_dark/"
img_utils.make_dir(sorted_dir)

for img_path in png_file_paths:
    file_name = img_path.split("/")[-1]
    
    if np.array(Image.open(img_path)).mean() > magic_img_mean - 10:
        os.system(f"cp {img_path} {sorted_dir}{file_name}")
