import matplotlib.pyplot as plt
from random import randint
import numpy as np
from PIL import Image
import wandb

from libs.img_utils import ImageUtils

iu = ImageUtils()

root_path = "/home/ladvien/deep_upscale/"
train_dir = f"{root_path}images/"
rich_file_paths = iu.get_image_files_recursively(train_dir + "rich/")

def get_batch(file_paths, batch_size, verbose=0):

    # Batches to return.
    rich_batch = []
    poor_batch = []

    # Get random files
    file_nums_to_load = [randint(0, len(file_paths)) - 1 for x in range(batch_size)]

    # Loop through the batch size, loading files.
    for i in range(batch_size):

        # Create matching file paths.
        rich_image_file_path = file_paths[file_nums_to_load[i]]
        poor_image_file_path = file_paths[file_nums_to_load[i]].replace(
            "/rich", "/poor"
        )
        if verbose > 0:
            print(f"Loading rich file: {rich_image_file_path}")
            print(f"Loading poor file: {poor_image_file_path}")

        # Load rich image, convert to B&W, and put into training batch.
        rich_image = Image.open(rich_image_file_path).convert("1")
        rich_batch.append(np.array(rich_image, dtype=int))

        # Load poor image, convert to B&W, and put into training batch.
        poor_image = Image.open(poor_image_file_path).convert("1")

        poor_batch.append(np.array(poor_image, dtype=int))
        print(np.array(poor_batch).shape)    
    return (np.array(poor_batch), np.array(rich_batch))

get_batch(rich_file_paths, 4, 1)