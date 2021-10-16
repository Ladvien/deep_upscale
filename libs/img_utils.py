import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2

import numpy as np

import os
import sys
from glob import glob

import matplotlib.pyplot as plt


class ImageUtils:
    def __init__(
        self,
    ):
        pass

    def find_subimage_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        return cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def find_subimages(self, image, minimum_size, verbose=0):

        images = []

        # Find contours
        cnts = self.find_subimage_contours(image)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)

            # # Skip if too small.
            if w < minimum_size or h < minimum_size:
                continue

            if verbose > 0:
                print(f"Found image -- W: {w} H: {h}")

            images.append(image[y : y + h, x : x + w])

        return images

    def save_subimages(
        self, filename, image_path, output_path, minimum_size, verbose=0
    ):

        image = cv2.imread(image_path, minimum_size)
        images = self.find_subimages(image, minimum_size, verbose)

        for i, image in enumerate(images):
            write_path = f"{output_path}/{filename}_{i}.png"
            print(f"Saving: {write_path}")
            cv2.imwrite(write_path, image)

    def midpoint(self, img):
        maxf = maximum_filter(img, (3, 3))
        minf = minimum_filter(img, (3, 3))
        return (maxf + minf) / 2

    def contraharmonic_mean(self, img, size, Q):
        num = np.power(img, Q + 1)
        denom = np.power(img, Q)
        kernel = np.full(size, 1.0)
        return cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)

    def get_image_files_recursively(self, root_dir, exclude_files=[]):
        file_types = ("*.jpg", "*.jpeg", "*.png")
        files = []

        for file_type in file_types:
            for dir, _, _ in os.walk(root_dir):
                print(dir)
                files.extend(glob(os.path.join(dir, file_type)))

        files = [file for file in files if file not in exclude_files]

        return files

    def remove_empty_images(self, image, threshold):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(image, axis=0)

        avg_color = avg_color.mean()

        # Invert the image if the average darkness is below
        # threshold.
        if avg_color < threshold:
            return image

        return None

    def show_pil_image(self, image, title):
        plt.title(title)
        plt.imshow(np.array(image), cmap="gray")
        plt.axis("off")
        plt.show()

    def convert_pil2cv2(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    ###################
    # Make directories
    ###################
    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def prepend_category_name_to_files(
        self, category_name: str, directory: str, input_path
    ):
        for count, filename in enumerate((os.listdir(directory))):
            dest_path = f"{input_path}{category_name}_{filename}"
            target_path = directory + filename
            os.system(f"mv {target_path} {dest_path}")
