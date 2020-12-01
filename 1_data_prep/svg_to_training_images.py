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


###########
# Imports
###########
import os
import sys
import glob

import numpy as np
import cairosvg
from PIL import Image

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

###############
# Setup Inputs
###############

tmp_folder = f"{project_path}images/tmp/"
if not os.path.exists(input_path):
    os.makedirs(input_path)
    
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

font_awesome_dir = f"{tmp_folder}font_awesome/black/svg/"
bootstrap_dir = f"{tmp_folder}bootstrap-icons-1.1.0/"
feather_dir = f"{tmp_folder}feather/icons/"

###############
# Get Data
###############
# Font Awesome
os.system(f"git clone https://github.com/encharm/Font-Awesome-SVG-PNG.git {tmp_folder}font_awesome")

# Bootstrap
os.system(f"wget https://github.com/twbs/icons/releases/download/v1.1.0/bootstrap-icons-1.1.0.zip --directory-prefix={tmp_folder}")
os.system(f"unzip {tmp_folder}*.zip -d {tmp_folder}")

# Feather
os.system(f"git clone https://github.com/feathericons/feather.git {tmp_folder}feather")



#####################
# Prevent Overwrites
#####################
os.chdir(font_awesome_dir)
os.system("rename 's/^/fontawesome/' *")

os.chdir(bootstrap_dir)
os.system("rename 's/^/bootstrap/' *")

os.chdir(feather_dir)
os.system("rename 's/^/feather/' *")

os.system(f"cp {font_awesome_dir}*.svg {input_path}")
os.system(f"cp {bootstrap_dir}*.svg {input_path} ")
os.system(f"cp {feather_dir}*.svg {input_path} ")


###################
# Helper Functions
###################
def show_pil_image(image, title):
    plt.title(title)
    plt.imshow(np.array(image), cmap="gray")
    plt.axis("off")
    plt.show()    

###################
# Make directories
###################
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


make_dir(poor_images_path)
make_dir(rich_images_path)

#################
# Get File Paths
#################
os.chdir(input_path)
svg_file_paths = []
for file in glob.glob("*.svg"):
    svg_file_paths.append(input_path + "/" + file)

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
        show_pil_image(poor_image, "Poor")
        show_pil_image(rich_image, "rich")