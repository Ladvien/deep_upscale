#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:37:35 2020

@author: ladvien
"""

"""
Downlaod Font-Awesome
git clone https://github.com/encharm/Font-Awesome-SVG-PNG
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

import matplotlib.pyplot as plt

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

poor_width = 128
poor_height = 128
poor_dims = (poor_width, poor_height)

x_small_factor_width = 3
x_small_factor_height = 3

rich_width = 128
rich_height = 128

samples_to_display = 10

# Noise
threshold               = 240
color_range             = 30
shape_range             = 15
size_range              = 1
num_pepper              = 5
specks_per_pepper       = 4
group_range             = 3

just_sample             = True

setup_data              = True


###############
# Setup Inputs
###############

tmp_folder = f"{project_path}images/tmp/"

if not os.path.exists(input_path):
    os.makedirs(input_path)
    
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
    os.system(f"git clone https://github.com/encharm/Font-Awesome-SVG-PNG.git {tmp_folder}font_awesome")
    
    # Bootstrap
    os.system(f"wget https://github.com/twbs/icons/releases/download/v1.1.0/bootstrap-icons-1.1.0.zip --directory-prefix={tmp_folder}")
    os.system(f"unzip {tmp_folder}*.zip -d {tmp_folder}")
    
    # Feather
    os.system(f"git clone https://github.com/feathericons/feather.git {tmp_folder}feather")
    
    # Hero
    os.system(f"git clone https://github.com/tailwindlabs/heroicons.git {tmp_folder}hero")

    # Ionicons
    os.system(f"git clone https://github.com/ionic-team/ionicons.git {tmp_folder}ionicons/")


#####################
# Prevent Overwrites
#####################

def prepend_category_name_to_files(category_name: str, directory: str):
    for count, filename in enumerate((os.listdir(directory))):
        print(filename)

prepend_category_name_to_files("test", font_awesome_dir)

# if setup_data:
#     os.chdir(font_awesome_dir)
#     os.system("rename 's/^/fontawesome_/' *")
    
#     os.chdir(bootstrap_dir)
#     os.system("rename 's/^/bootstrap_/' *")
    
#     os.chdir(feather_dir)
#     os.system("rename 's/^/feather_/' *")
    
#     os.chdir(hero_solid_dir)
#     os.system("rename 's/^/solid_hero_/' *")    
    
#     os.chdir(hero_outline_dir)
#     os.system("rename 's/^/outline_hero_/' *")
    
#     os.chdir(ionicons_dir)
#     os.system("rename 's/^/ionicons_/' *")
    
#     os.system(f"cp {font_awesome_dir}*.svg {input_path}")
#     os.system(f"cp {bootstrap_dir}*.svg {input_path} ")
#     os.system(f"cp {feather_dir}*.svg {input_path} ")
#     os.system(f"cp {hero_solid_dir}*.svg {input_path} ")
#     os.system(f"cp {hero_outline_dir}*.svg {input_path} ")
#     os.system(f"cp {ionicons_dir}*.svg {input_path}")
    
#     os.chdir(f"{project_path}")


# ###################
# # Helper Functions
# ###################
# def show_pil_image(image, title):
#     plt.title(title)
#     plt.imshow(np.array(image), cmap="gray")
#     plt.axis("off")
#     plt.show()    

# def convert_pil2cv2(pil_image):
#     return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# def add_noise(image, num_pepper, shape_range, color_range, specks_per_pepper, group_range, size_range):
#     noise_img = image.copy()
#     for _ in range(num_pepper):
#         # Radius of circle
#         radius = randint(0, shape_range)

#         b = randint(0, color_range)
#         g = randint(0, color_range)
#         r = randint(0, color_range)

#         # BGR
#         color = (b, g, r)

#         # Center coordinates
#         y = randint(0, image.size[0])
#         x = randint(0, image.size[1])

#         for j in range(specks_per_pepper):        

#             group_x_offset = randint(group_range*-1, group_range)
#             group_y_offset = randint(group_range*-1, group_range)

#             # Size
#             radius = randint(0, size_range)

#             # Convert from PIL to cv2.
#             noise_img = convert_pil2cv2(noise_img)

#             # Add noise.
#             noise_img = cv2.circle(noise_img, (x + group_x_offset, y + group_y_offset), radius, color, -1)

#     # Return PIL image.
#     return Image.fromarray(noise_img)

# ###################
# # Make directories
# ###################
# def make_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


# make_dir(poor_images_path)
# make_dir(rich_images_path)

# #################
# # Get File Paths
# #################
# os.chdir(input_path)
# svg_file_paths = [input_path + "/" + file for file in glob.glob("*.svg")]

# ###################
# # Generate Images
# ###################

# index = 0
# for file_path in svg_file_paths:

#     # Get image file name.
#     file_name = file_path.split("/")[-1].replace(".svg", ".png")

#     # Paths
#     poor_file_path = poor_images_path + file_name
#     rich_file_path = rich_images_path + file_name

#     # Save really small.    
#     cairosvg.svg2png(
#         url=file_path,
#         write_to=poor_file_path,
#         output_width=round(poor_width / x_small_factor_width),
#         output_height=round(poor_width / x_small_factor_height),
#         background_color="ghostwhite",
#     )

#     # Load the really small
#     xs_image = Image.open(poor_file_path)
#     # Make it target size for poor image.
#     poor_image = xs_image.resize(((poor_dims)))
#     poor_image = add_noise(poor_image, num_pepper, shape_range, color_range, specks_per_pepper, group_range, size_range)


#     # Save it again.
#     poor_image.save(poor_file_path)


#     # Save really rich.    
#     cairosvg.svg2png(
#         url=file_path,
#         write_to=rich_file_path,
#         output_width=rich_width,
#         output_height=rich_height,
#         background_color="ghostwhite",
#     )
#     rich_image = Image.open(rich_file_path)

#     index += 1

#     # Show what what's going on.
#     if index < samples_to_display:
#         show_pil_image(poor_image, "Poor")
#         show_pil_image(rich_image, "Rich")
#     elif just_sample:
#         break