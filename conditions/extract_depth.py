import torch
import numpy as np
import PIL.Image as PImage
import os
import cv2
import shutil
from util import resize_image, HWC3, load_image
from midas import MidasDetector

apply_midas = MidasDetector()


def process(input_image, detect_resolution=384, bg_threshold=0.4):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
        detected_map = HWC3(detected_map)
        H, W, C = input_image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = PImage.fromarray(detected_map)

        return detected_map


def generate_condition(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        print(root)
        for file in files:
            if file.lower().endswith('.jpeg'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                condition = process(np.array(load_image(file_path)))
                condition.save(os.path.join(output_path, file))

    print("finish")


input_directory = None
output_directory = None
generate_condition(input_directory, output_directory)
