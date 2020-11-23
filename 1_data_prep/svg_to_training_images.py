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
# Get data
# !git clone https://github.com/encharm/Font-Awesome-SVG-PNG.git

# Install CairoSVG
# Installation instructions.
# !pip install cairosvg


###########
# Imports
###########
import os
import sys
import glob

import cairosvg


#############
# Setup
#############
root_path = os.environ["HOME"]


#################
# Parameters
#################
input_path              = f"{root_path}/deep_upscale/Font-Awesome-SVG-PNG/black/svg"
output_path             = f"{root_path}/deep_upscalae/images/"

poor_images_path        = f"{output_path}poor/"
rich_images_path        = f"{output_path}rich/"

poor_width           = 64 
poor_height          = 64

rich_width           = 512 
rich_height          = 512

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
    
for file_path in svg_file_paths:
    cairosvg.svg2png(url=file_path, write_to=output_path)