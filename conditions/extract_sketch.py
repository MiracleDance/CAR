import torch
import numpy as np
import PIL.Image as PImage
import os
import cv2
import shutil
from util import resize_image, HWC3, load_image, nms, ckpt_path

model_pidinet = None
device = 'cuda:0'


def pidinet(img, res=512, **kwargs):
    img = resize_image(img, res)
    global model_pidinet
    if model_pidinet is None:
        from pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, device=device, ckpt_path=ckpt_path)
    return result


def scribble_pidinet(img, res=512, **kwargs):
    result = pidinet(img, res)
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result


def generate_condition(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        print(root)
        for file in files:
            if file.lower().endswith('.jpeg'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                image = cv2.imread(file_path)
                condition = scribble_pidinet(image)
                cv2.imwrite(os.path.join(output_path, file), condition)

    print("finish")


input_directory = None
output_directory = None
generate_condition(input_directory, output_directory)
