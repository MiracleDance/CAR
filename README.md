<h1> CAR<img src="./docs/car.png" width="4%">: Controllable AutoRegressive Modeling for Visual Generation </h1>

Ziyu Yao<sup>1,2</sup>, Jialin Li<sup>2</sup>, Yifeng Zhou<sup>2</sup>, Yong Liu<sup>2</sup>, Xi Jiang<sup>2,3</sup>, Chengjie Wang<sup>2</sup>, Feng Zheng<sup>3</sup>, Yuexian Zou<sup>1</sup>, Lei Li<sup>4</sup>

<sup>1</sup> Peking University,
<sup>2</sup> Tencent Youtu Lab,
<sup>3</sup> Southern University of Science and Technology,
<sup>4</sup> University of Washington

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.04671-b31b1b.svg)](https://arxiv.org/abs/2410.04671)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-MiracleDance/CAR-yellow)](https://huggingface.co/MiracleDance/CAR)&nbsp;

</div>

<div align="center">
<img src="./docs/teaser.png" width="80%">
</div>

## CAR Models
We have currently released the CAR-d16 weights for demo purposes, and larger models will be made available following future upgrades and extensions of CAR.

The CAR models are available on <a href='https://huggingface.co/MiracleDance/CAR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-MiracleDance/CAR-yellow'></a> and can also be downloaded from the following links:

|   Model    | reso. |  Condition  | HF weights🤗                                                                                   |
|:----------:|:-----:|:-----------:|:-----------------------------------------------------------------------------------------------|
|  CAR-d16   |  256  |  Canny Edge | [car_canny_d16.pth](https://huggingface.co/MiracleDance/CAR/resolve/main/car_canny_d16.pth)    |
|  CAR-d16   |  256  |  HED Map    | [car_hed_d16.pth](https://huggingface.co/MiracleDance/CAR/resolve/main/car_hed_d16.pth)        |
|  CAR-d16   |  256  |  Depth Map  | [car_depth_d16.pth](https://huggingface.co/MiracleDance/CAR/resolve/main/car_depth_d16.pth)    |
|  CAR-d16   |  256  |  Normal Map | [car_normal_d16.pth](https://huggingface.co/MiracleDance/CAR/resolve/main/car_normal_d16.pth)  |
|  CAR-d16   |  256  |   Sketch    | [car_sketch_d16.pth](https://huggingface.co/MiracleDance/CAR/resolve/main/car_sketch_d16.pth)  |

As CAR is based on the pre-trained [VAR](https://github.com/FoundationVision/VAR) model, the following pre-trained weights also need to be downloaded: [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth), [var_d16.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth).

## Training
#### 1. Prepare Dataset
The arg `--data_path` should indicate the path to the [ImageNet](http://image-net.org/) dataset.

#### 2. Extract conditions from ImageNet dataset
You can choose to extract conditions from all categories or select a subset of 1000 categories for condition extraction. Run the following commands:
```shell
# canny
python extract_canny.py
# hed
python extract_hed.py
# depth
python extract_depth.py
# normal
python extract_normal.py
# sketch
python extract_sketch.py
```

#### 3. Train CAR model
```shell
# d16, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --data_path=/path/to/imagenet --condition_path=/path/to/condition/extract/above \
  --vae_ckpt=/path/to/pretrained/vae/ckpt --pretrained_var_ckpt=/path/to/pretrained/var/ckpt \
  --tblr=0.0001 --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 
```

## Inference
```shell
# cls is an index ranging from 0 to 999 in the ImageNet label set
# type indicates which condition is extracted from the original image (canny, hed, depth, normal, sketch)
python inference.py --vae_ckpt=/path/to/pretrained/vae/ckpt --var_ckpt=/path/to/pretrained/var/ckpt \
  --car_ckpt=/path/to/car/ckpt --img_path=/path/to/original/image/to/extract/condition \
  --save_path=/path/to/save/image --cls=3 --type=hed
```

## Acknowledgments
The development of CAR is based on [VAR](https://github.com/FoundationVision/VAR). We deeply appreciate this significant contribution to the community.

## Citation
If you find our work helpful in your research, we would be grateful if you could consider giving us a star ⭐ or citing it using:
```
@article{yao2024car,
  title={Car: Controllable autoregressive modeling for visual generation},
  author={Yao, Ziyu and Li, Jialin and Zhou, Yifeng and Liu, Yong and Jiang, Xi and Wang, Chengjie and Zheng, Feng and Zou, Yuexian and Li, Lei},
  journal={arXiv preprint arXiv:2410.04671},
  year={2024}
}
```
