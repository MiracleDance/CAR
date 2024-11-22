import torch

best_ckpt = '/data/yzy/projects/conVAR_residual/local_output_car_hed_lr0.001/ar-ckpt-best.pth'
a = torch.load(best_ckpt, map_location='cpu')['trainer']['car_wo_ddp']
b = {}
for k, v in a.items():
    if 'car' in k:
        b[k] = v
torch.save(b, '/data/yzy/projects/conVAR_residual/local_output_car_hed_lr0.001/ckpt-best.pth')