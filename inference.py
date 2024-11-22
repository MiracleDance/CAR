import os
import argparse
import torch
import torchvision
import random
import numpy as np
import cv2
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from models import VQVAE, build_car
from utils.loading_utils import load_image
from utils.control_data_utils import pil_to_numpy, numpy_to_pt
from utils.extract_control import extract


def main(args):
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

    MODEL_DEPTH = 16
    assert MODEL_DEPTH in {16, 20, 24, 30}

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

    vae, car = build_car(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)

    var_weights = torch.load(args.var_ckpt, map_location='cpu')
    car_weights = torch.load(args.car_ckpt, map_location='cpu')
    all_weights = {}
    all_weights.update(var_weights)
    all_weights.update(car_weights)
    car.load_state_dict(all_weights, strict=True)

    vae.eval(), car.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in car.parameters():
        p.requires_grad_(False)

    seed = 100
    torch.manual_seed(seed)
    num_sampling_steps = 250
    cfg = 4
    more_smooth = False

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    def get_control_for_each_scale(control_image, scale):
        def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
            return x.add(x).add_(-1)
        c_tensors = []
        c_images = []
        for pn in scale:
            c_res = control_image.resize((pn * 16, pn * 16))
            c_images.append(c_res)
            c_tensors.append(normalize_01_into_pm1(numpy_to_pt(pil_to_numpy(c_res))))
        return c_images, c_tensors

    control = extract(args.type, args.img_path)

    control_images, control_tensors = get_control_for_each_scale(control, patch_nums)
    class_labels = [args.cls] * 8
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    for i in range(len(control_tensors)):
        control_tensors[i] = control_tensors[i].repeat(B, 1, 1, 1).to(device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):  # using bfloat16 can be faster
            recon_B3HW = car.car_inference(B=B, label_B=label_B, cfg=cfg,
                                           top_k=900, top_p=0.95, g_seed=seed,
                                           more_smooth=more_smooth, control_tensors=control_tensors)

    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))

    combined_width = control.width + chw.width
    combined_height = control.height
    combined = PImage.new('RGB', (combined_width, combined_height))

    combined.paste(control, (0, 0))
    combined.paste(chw, (control.width, 0))
    combined.save(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAR Inference Single Image')
    parser.add_argument('--vae_ckpt', type=str, help='path to the pre-trained vae checkpoint')
    parser.add_argument('--var_ckpt', type=str, help='path to the pre-trained var checkpoint')
    parser.add_argument('--car_ckpt', type=str, help='path to the car checkpoint')
    parser.add_argument('--img_path', type=str, help='path to an original image from which the condition is extracted')
    parser.add_argument('--save_path', type=str, help='path to the generated image')
    parser.add_argument('--cls', type=int, help='an index ranging from 0 to 999 in the ImageNet label set')
    # parser.add_argument('--type', type=str, help='indicating which condition is extracted from the original image')
    args = parser.parse_args()
    args.type = 'hed'  # currently, hed_map only
    main(args)
