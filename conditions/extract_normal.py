import torch
import numpy as np
import PIL.Image as PImage
import os
import cv2
import shutil
from util import resize_image, HWC3, load_image
from normalbae import NormalBaeDetector


normal_bae = NormalBaeDetector.from_pretrained("ckpts").to('cuda:0')


def generate_condition(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        print(root)
        for file in files:
            if file.lower().endswith('.jpeg'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                condition = PImage.fromarray(normal_bae(np.array(load_image(file_path), dtype=np.uint8), output_type="np"))
                # condition = process(np.array(load_image(file_path)))
                condition.save(os.path.join(output_path, file))

    print("finish")


input_directory = None
output_directory = None
generate_condition(input_directory, output_directory)
