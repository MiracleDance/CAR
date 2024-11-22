import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from models.var import VAR


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class ControlConditionEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class CAR(VAR):
    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        super(CAR, self).__init__(vae_local, num_classes, depth, embed_dim, num_heads, mlp_ratio,
                                  drop_rate, attn_drop_rate, drop_path_rate, norm_eps, shared_aln,
                                  cond_drop_rate, attn_l2_norm, patch_nums, flash_if_available, fused_if_available)

        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.car_control_convs = ControlConditionEmbedding(conditioning_embedding_channels=self.C)
        self.car_var_conv = nn.Conv2d(self.C, self.C, kernel_size=conv_in_kernel, padding=conv_in_padding)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.car_blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth // 2)
        ])

        car_norm_layer = FP32_Layernorm
        car_skip_norm = []
        car_skip_linear = []
        for _ in range(depth // 2):
            car_skip_norm.append(car_norm_layer(2 * self.C, elementwise_affine=True, eps=1e-6))
            car_skip_linear.append(nn.Linear(2 * self.C, self.C))
        self.car_skip_norm = nn.ModuleList(car_skip_norm)
        self.car_skip_linear = nn.ModuleList(car_skip_linear)

    @torch.no_grad()
    def car_inference(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, control_tensors=None
    ) -> torch.Tensor:
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        control_f = []
        if control_tensors is not None:
            assert control_tensors[0].shape[0] == B
            for control_tensor in control_tensors:
                control_i = self.car_control_convs(control_tensor)
                control_f.append(control_i)

        for cb in self.car_blocks:
            cb.attn.kv_caching(True)

        next_control_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)

        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map

            control_residual_f = []
            if control_tensors is not None:
                control_x = control_f[si].repeat(2, 1, 1, 1)
                var_x = next_control_token_map.transpose(1, 2).contiguous().reshape(2 * B, self.C, pn, pn)
                var_x = self.car_var_conv(var_x)
                control_x = var_x + control_x
                control_x = control_x.view(2 * B, self.C, -1).transpose(1, 2)
                control_x = control_x + lvl_pos[:, cur_L - pn * pn: cur_L]

                for cb in self.car_blocks:
                    control_x = cb(x=control_x, cond_BD=cond_BD_or_gss, attn_bias=None)
                    control_residual_f.append(control_x)

            for bidx, b in enumerate(self.blocks):
                if control_tensors is not None and bidx >= len(self.blocks) // 2:
                    con_f = control_residual_f.pop()
                    cat = torch.cat([x, con_f], dim=-1)
                    cat = self.car_skip_norm[bidx - len(self.blocks) // 2](cat)
                    x = self.car_skip_linear[bidx - len(self.blocks) // 2](cat)
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_control_token_map = self.word_embed(next_token_map).repeat(2, 1, 1)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

        for b in self.blocks:
            b.attn.kv_caching(False)

        for cb in self.car_blocks:
            cb.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor,
                control_tensors=None) -> torch.Tensor:  # returns logits_BLV

        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l[0].shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            control_f = []
            if control_tensors is not None:
                assert control_tensors[0].shape[0] == B
                for control_tensor in control_tensors:
                    control_i = self.car_control_convs(control_tensor)
                    control_f.append(control_i)

            car_input = []
            var_x = sos.transpose(1, 2).contiguous().reshape(B, self.C, self.patch_nums[0], self.patch_nums[0])
            var_x = self.car_var_conv(var_x)
            car_x = var_x + control_f[0]
            car_x = car_x.view(B, self.C, -1).transpose(1, 2).contiguous()
            car_input.append(car_x)
            for si, (pn, var_input) in enumerate(zip(self.patch_nums[1:], x_BLCv_wo_first_l)):
                var_x = self.word_embed(var_input.float())
                var_x = var_x.transpose(1, 2).contiguous().reshape(B, self.C, pn, pn)
                var_x = self.car_var_conv(var_x)
                car_x = var_x + control_f[si + 1]
                car_x = car_x.view(B, self.C, -1).transpose(1, 2).contiguous()
                car_input.append(car_x)

            car_input = torch.cat(car_input, dim=1)
            car_input += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

            x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=1)

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        control_residual_f = []
        for cb in self.car_blocks:
            car_input = cb(x=car_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            control_residual_f.append(car_input)

        for i, b in enumerate(self.blocks):
            if i >= len(self.blocks) // 2:
                con_f = control_residual_f.pop()
                cat = torch.cat([x_BLC, con_f], dim=-1)
                cat = self.car_skip_norm[i - len(self.blocks) // 2](cat)
                x_BLC = self.car_skip_linear[i - len(self.blocks) // 2](cat)
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        return x_BLC  # logits BLV, V is vocab_size
