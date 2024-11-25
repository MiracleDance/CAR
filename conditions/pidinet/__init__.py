import os
import torch
import numpy as np
from einops import rearrange
from ..pidinet.model import pidinet
from ..util import load_state_dict

netNetwork = None


def apply_pidinet(input_image, device, ckpt_path, apply_fliter=False):
    global netNetwork
    if netNetwork is None:
        # please download: https://huggingface.co/lllyasviel/Annotators/blob/main/table5_pidinet.pth
        model_path = os.path.join(ckpt_path, 'table5_pidinet.pth')
        assert os.path.exists(model_path)
        netNetwork = pidinet()
        ckp = load_state_dict(model_path)
        netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()})

    netNetwork = netNetwork.to(device)
    netNetwork.eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(device)
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = netNetwork(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_fliter:
            edge = edge > 0.5
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

    return edge[0][0]
