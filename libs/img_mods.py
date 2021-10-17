from random import randint
from libs.img_utils import ImageUtils
import cv2
from PIL import Image
import numpy as np

img_utils = ImageUtils()


def add_noise(
    image,
    num_pepper,
    shape_range,
    color_range,
    specks_per_pepper,
    group_range,
    size_range,
):
    noise_img = image.copy()
    for _ in range(num_pepper):
        # Radius of circle
        radius = randint(0, shape_range)

        b = randint(0, color_range)
        g = randint(0, color_range)
        r = randint(0, color_range)

        # BGR
        color = (b, g, r)

        # Center coordinates
        y = randint(0, image.size[0])
        x = randint(0, image.size[1])

        for j in range(specks_per_pepper):

            group_x_offset = randint(group_range * -1, group_range)
            group_y_offset = randint(group_range * -1, group_range)

            # Size
            radius = randint(0, size_range)

            # Convert from PIL to cv2.
            noise_img = img_utils.convert_pil2cv2(noise_img)

            # Add noise.
            noise_img = cv2.circle(
                noise_img, (x + group_x_offset, y + group_y_offset), radius, color, -1
            )

    # Return PIL image.
    return Image.fromarray(noise_img)


def convert_image_to_bw(image, threshold):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 255, threshold, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    thresh, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image


def invert_mostly_black_images(image, threshold):

    try:
        image = img_utils.convert_pil2cv2(image)
    except Exception as e:
        print(e)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(image, axis=0)

    avg_color = avg_color.mean()

    # Invert the image if the average darkness is below
    # threshold.
    if avg_color < threshold:
        image = ~image
    return image
