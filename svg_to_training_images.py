#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:37:35 2020

@author: ladvien
"""

################
# Pre Execution
################

# !pip install numpy
# !pip install cairosvg
# !pip install opencv-python
# !pip install matplotlib
# !pip install pillow

###########
# Imports
###########
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

just_sample = True
setup_data = True


poor_width = 128
poor_height = 128
poor_dims = (poor_width, poor_height)

x_small_factor_width = 3
x_small_factor_height = 3

rich_width = 128
rich_height = 128

samples_to_display = 10

# Noise
threshold = 240
color_range = 30
shape_range = 15
size_range = 1
num_pepper = 5
specks_per_pepper = 4
group_range = 3



###############
# Setup Inputs
###############

tmp_folder = f"{project_path}images/tmp/"

img_utils.make_dir(input_path)

if os.path.exists(tmp_folder):
    os.system(f"rm -rf {tmp_folder}")
os.makedirs(tmp_folder)

font_awesome_dir = f"{tmp_folder}font_awesome/black/svg/"
bootstrap_dir = f"{tmp_folder}bootstrap-icons-1.1.0/"
feather_dir = f"{tmp_folder}feather/icons/"
hero_solid_dir = f"{tmp_folder}hero/src/solid/"
hero_outline_dir = f"{tmp_folder}hero/src/outline/"
ionicons_dir = f"{tmp_folder}ionicons/src/svg/"


###############
# Get Data
###############

if setup_data:
    # Font Awesome
    os.system(
        f"git clone https://github.com/encharm/Font-Awesome-SVG-PNG.git {tmp_folder}font_awesome"
    )

    # Bootstrap
    os.system(
        f"wget https://github.com/twbs/icons/releases/download/v1.1.0/bootstrap-icons-1.1.0.zip --directory-prefix={tmp_folder}"
    )
    os.system(f"unzip {tmp_folder}*.zip -d {tmp_folder}")

    # Feather
    os.system(
        f"git clone https://github.com/feathericons/feather.git {tmp_folder}feather"
    )

    # Hero
    os.system(
        f"git clone https://github.com/tailwindlabs/heroicons.git {tmp_folder}hero"
    )

    # Ionicons
    os.system(
        f"git clone https://github.com/ionic-team/ionicons.git {tmp_folder}ionicons/"
    )


#####################
# Prevent Overwrites
#####################

if setup_data:
    img_utils.prepend_category_name_to_files("font_awesome", font_awesome_dir, input_path)
    img_utils.prepend_category_name_to_files("bootstrap_dir", bootstrap_dir, input_path)
    img_utils.prepend_category_name_to_files("feather_dir", feather_dir, input_path)
    img_utils.prepend_category_name_to_files("hero_solid_dir", hero_solid_dir, input_path)
    img_utils.prepend_category_name_to_files("hero_outline_dir", hero_outline_dir, input_path)
    img_utils.prepend_category_name_to_files("ionicons_dir", ionicons_dir, input_path)
    img_utils.prepend_category_name_to_files("bootstrap_dir", bootstrap_dir, input_path)


img_utils.make_dir(poor_images_path)
img_utils.make_dir(rich_images_path)

# #################
# # Get File Paths
# #################
os.chdir(input_path)
svg_file_paths = [input_path + "/" + file for file in glob.glob("*.svg")]

###################
# Generate Images
###################

index = 0
for file_path in svg_file_paths:

    # Get image file name.
    file_name = file_path.split("/")[-1].replace(".svg", ".png")

    # Paths
    poor_file_path = poor_images_path + file_name
    rich_file_path = rich_images_path + file_name

    # Save really small.
    cairosvg.svg2png(
        url=file_path,
        write_to=poor_file_path,
        output_width=round(poor_width / x_small_factor_width),
        output_height=round(poor_width / x_small_factor_height),
        background_color="ghostwhite",
    )

    # Load the really small
    xs_image = Image.open(poor_file_path)
    # Make it target size for poor image.
    poor_image = xs_image.resize(((poor_dims)))
    poor_image = add_noise(
        poor_image,
        num_pepper,
        shape_range,
        color_range,
        specks_per_pepper,
        group_range,
        size_range,
    )

    # Save it again.
    poor_image.save(poor_file_path)

    # Save really rich.
    cairosvg.svg2png(
        url=file_path,
        write_to=rich_file_path,
        output_width=rich_width,
        output_height=rich_height,
        background_color="ghostwhite",
    )
    rich_image = Image.open(rich_file_path)

    index += 1

    # Show what what's going on.
    if index < samples_to_display:
        img_utils.show_pil_image(poor_image, "Poor")
        img_utils.show_pil_image(rich_image, "Rich")
    elif just_sample:
        break
