import torch
import numpy as np
import PIL.Image as PImage
import os
import cv2
import shutil
from util import resize_image, HWC3, load_image


def generate_condition(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        print(root)
        for file in files:
            if file.lower().endswith('.jpeg'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                input_image = load_image(file_path)
                canny_image = np.array(input_image.resize(size=(256, 256)))
                low_threshold = 100
                high_threshold = 200
                canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
                W, H = input_image.size
                # canny_image = cv2.resize(canny_image, (W, H), interpolation=cv2.INTER_LINEAR)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                canny_image = PImage.fromarray(canny_image)
                canny_image = canny_image.resize(size=(W, H))

                canny_image.save(os.path.join(output_path, file))

    print("finish")


input_directory = None  # imagenet dir path
output_directory = None  # condition save path
generate_condition(input_directory, output_directory)
