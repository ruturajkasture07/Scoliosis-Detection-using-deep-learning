"""
VerteNet Inference — exact architecture from the training notebook.
===================================================================
Cell 1 of your Colab notebook should be:
    %%writefile vertenet_infer.py
    <this entire file>

Then Cell 2:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "einops"], check=True)
    if 'vertenet_infer' in sys.modules:
        del sys.modules['vertenet_infer']
    from vertenet_infer import VerteNetInference
    print("loaded")

Then Cell 3:
    WEIGHTS = "exp1_baseline_phase1_best.pth"
    IMAGE   = "spine.jpg"
    model  = VerteNetInference(weights_path=WEIGHTS, mode='strip')
    result = model.visualize(IMAGE, out_dir="./output")
    print(result['cobb_angles'])
"""

import os, sys, json, warnings, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

warnings.filterwarnings('ignore')


# =============================================================================
# 1.  MODEL PARTS — verbatim from notebook Cell "model-parts"
#     The REAL CombinationModule_proposed uses patch-based Linear attention,
#     MaxPool2d downsampling, and ConvTranspose2d — NOT Conv2d attention.
#     This is the version that was actually saved into your .pth checkpoint.
# =============================================================================

# ── patch helpers ─────────────────────────────────────────────────────────────
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def get_patches(image, patch_size):
    image = rearrange(image, 'b c h w -> b h w c')
    b, h, w, c = image.shape
    return rearrange(image, 'b (h_p h) (w_p w) c -> b (h_p w_p) h w c',
                     h_p=patch_size[0], w_p=patch_size[1],
                     h=h // patch_size[0], w=w // patch_size[1])

def reconstruct_image(patches, image_shape, patch_size):
    b, h, w, c = image_shape
    image = rearrange(patches, 'b (h_p w_p) h w c -> b (h_p h) (w_p w) c',
                      h_p=patch_size[0], w_p=patch_size[1],
                      h=h // patch_size[0], w=w // patch_size[1])
    return rearrange(image, 'b h w c -> b c h w')


# ── Layer norms ───────────────────────────────────────────────────────────────
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        return x / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias   = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5) \
               * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' \
                    else WithBias_LayerNorm(dim)
    def forward(self, x):
        return self.body(x)


class LayerNorm_Channel(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' \
                    else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                     groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


# ── DR__CrossAttention — patch-based, verbatim from notebook ─────────────────
class DR__CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, channels, patch_size):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_size  = patch_size
        ps2 = patch_size * patch_size
        self.q_Hi = nn.Linear(ps2, ps2, bias=False)
        self.k_Hi = nn.Linear(ps2, ps2, bias=False)
        self.v_Hi = nn.Linear(ps2, ps2, bias=False)
        self.q_Lo = nn.Linear(ps2, ps2, bias=False)
        self.k_Lo = nn.Linear(ps2, ps2, bias=False)
        self.v_Lo = nn.Linear(ps2, ps2, bias=False)
        self.downsample    = nn.MaxPool2d(2, 2)
        self.convTranspose = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.project_out   = nn.Conv2d(channels * 2, channels, 1, bias=bias)

    def forward(self, x_up, x_down):
        ps = self.patch_size

        def patch_and_rearrange(x):
            b, c, h, w = x.shape
            p = get_patches(x, (ps, ps))
            _, _, ph, pw, _ = p.shape
            return rearrange(p, 'b pts h w c -> b (h w) c pts'), b, c, h, w, ph, pw

        pd_Hi, *shd_Hi = patch_and_rearrange(x_down)
        pd_Lo, *shd_Lo = patch_and_rearrange(self.downsample(x_down))
        pu_Hi, *shu_Hi = patch_and_rearrange(x_up)
        pu_Lo, *shu_Lo = patch_and_rearrange(self.downsample(x_up))

        q_dHi = rearrange(self.q_Hi(pd_Hi), 'b hw c p -> b hw p c')
        q_dLo = rearrange(self.q_Lo(pd_Lo), 'b hw c p -> b hw p c')
        k_uHi = rearrange(self.k_Hi(pu_Hi), 'b hw c p -> b hw p c')
        k_uLo = rearrange(self.k_Lo(pu_Lo), 'b hw c p -> b hw p c')
        v_uHi = rearrange(self.v_Hi(pu_Hi), 'b hw c p -> b hw p c')
        v_uLo = rearrange(self.v_Lo(pu_Lo), 'b hw c p -> b hw p c')

        q_dHi = F.normalize(q_dHi, -1); k_uHi = F.normalize(k_uHi, -1)
        q_dLo = F.normalize(q_dLo, -1); k_uLo = F.normalize(k_uLo, -1)

        out_Hi = (q_dHi @ k_uHi.transpose(-2, -1) * self.temperature).softmax(-1) @ v_uHi
        out_Lo = (q_dLo @ k_uLo.transpose(-2, -1) * self.temperature).softmax(-1) @ v_uLo

        b, c, h, w = x_down.shape
        out_Hi = rearrange(out_Hi, 'b hw p c -> b hw c p')
        out_Hi = rearrange(out_Hi, 'b (h w) c p -> b p h w c', h=shd_Hi[4], w=shd_Hi[5])
        out_Hi = reconstruct_image(out_Hi, (b, h, w, c), (ps, ps))

        bL, cL, hL, wL = self.downsample(x_down).shape
        out_Lo = rearrange(out_Lo, 'b hw p c -> b hw c p')
        out_Lo = rearrange(out_Lo, 'b (h w) c p -> b p h w c', h=shd_Lo[4], w=shd_Lo[5])
        out_Lo = reconstruct_image(out_Lo, (bL, hL, wL, cL), (ps, ps))
        out_Lo = self.convTranspose(out_Lo)

        return self.project_out(torch.cat([out_Hi, out_Lo], 1))


# ── DR__SelfAttention — patch-based, verbatim from notebook ──────────────────
class DR__SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, channels, patch_size):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_size  = patch_size
        ps2 = patch_size * patch_size
        self.q_Hi = nn.Linear(ps2, ps2, bias=False)
        self.k_Hi = nn.Linear(ps2, ps2, bias=False)
        self.v_Hi = nn.Linear(ps2, ps2, bias=False)
        self.q_Lo = nn.Linear(ps2, ps2, bias=False)
        self.k_Lo = nn.Linear(ps2, ps2, bias=False)
        self.v_Lo = nn.Linear(ps2, ps2, bias=False)
        self.downsample    = nn.MaxPool2d(2, 2)
        self.convTranspose = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.project_out   = nn.Conv2d(channels * 2, channels, 1, bias=bias)

    def forward(self, x):
        ps = self.patch_size
        b, c, h, w = x.shape
        x_Lo = self.downsample(x)

        def prep(t):
            tb, tc, th, tw = t.shape
            p = get_patches(t, (ps, ps))
            _, _, ph, pw, _ = p.shape
            return rearrange(p, 'b pts h w c -> b (h w) c pts'), tb, tc, th, tw, ph, pw

        p_Hi, *sh_Hi = prep(x)
        p_Lo, *sh_Lo = prep(x_Lo)

        q_Hi = rearrange(self.q_Hi(p_Hi), 'b hw c p -> b hw p c')
        k_Hi = rearrange(self.k_Hi(p_Hi), 'b hw c p -> b hw p c')
        v_Hi = rearrange(self.v_Hi(p_Hi), 'b hw c p -> b hw p c')
        q_Lo = rearrange(self.q_Lo(p_Lo), 'b hw c p -> b hw p c')
        k_Lo = rearrange(self.k_Lo(p_Lo), 'b hw c p -> b hw p c')
        v_Lo = rearrange(self.v_Lo(p_Lo), 'b hw c p -> b hw p c')

        q_Hi = F.normalize(q_Hi, -1); k_Hi = F.normalize(k_Hi, -1)
        q_Lo = F.normalize(q_Lo, -1); k_Lo = F.normalize(k_Lo, -1)

        out_Hi = (q_Hi @ k_Hi.transpose(-2, -1) * self.temperature).softmax(-1) @ v_Hi
        out_Lo = (q_Lo @ k_Lo.transpose(-2, -1) * self.temperature).softmax(-1) @ v_Lo

        out_Hi = rearrange(out_Hi, 'b hw p c -> b hw c p')
        out_Hi = rearrange(out_Hi, 'b (h w) c p -> b p h w c', h=sh_Hi[4], w=sh_Hi[5])
        out_Hi = reconstruct_image(out_Hi, (b, h, w, c), (ps, ps))

        bL, cL, hL, wL = x_Lo.shape
        out_Lo = rearrange(out_Lo, 'b hw p c -> b hw c p')
        out_Lo = rearrange(out_Lo, 'b (h w) c p -> b p h w c', h=sh_Lo[4], w=sh_Lo[5])
        out_Lo = reconstruct_image(out_Lo, (bL, hL, wL, cL), (ps, ps))
        out_Lo = self.convTranspose(out_Lo)

        return self.project_out(torch.cat([out_Hi, out_Lo], 1))


# ── TransformerBlock_proposed — verbatim from notebook ───────────────────────
class TransformerBlock_proposed(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type, channels, patch_size):
        super().__init__()
        self.norm1      = LayerNorm_Channel(channels, LayerNorm_type)
        self.norm1b     = LayerNorm_Channel(channels, LayerNorm_type)
        self.norm2      = LayerNorm_Channel(channels, LayerNorm_type)
        self.self_attn  = DR__SelfAttention(dim, num_heads, bias, channels, patch_size)
        self.cross_attn = DR__CrossAttention(dim, num_heads, bias, channels, patch_size)
        self.ffn        = FeedForward(channels, ffn_expansion_factor, bias)

    def forward(self, x_up, x_down):
        x_down = x_down + self.self_attn(self.norm1(x_down))
        x_down = x_down + self.cross_attn(self.norm1b(x_up), self.norm1b(x_down))
        x_down = x_down + self.ffn(self.norm2(x_down))
        return x_down


# ── CombinationModule_proposed — verbatim from notebook ──────────────────────
class CombinationModule_proposed(nn.Module):
    """VerteNet's novel combination: DR cross-attention + transformer block."""
    def __init__(self, c_low, c_up, batch_norm=True, patch_size=8, dim=None):
        super().__init__()
        if batch_norm:
            self.up = nn.Sequential(
                nn.Conv2d(c_low, c_up, 3, 1, 1),
                nn.BatchNorm2d(c_up),
                nn.ReLU(True))
        else:
            self.up = nn.Sequential(
                nn.Conv2d(c_low, c_up, 3, 1, 1),
                nn.ReLU(True))
        self.transformer = TransformerBlock_proposed(
            dim=c_up, num_heads=1, ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias',
            channels=c_up, patch_size=patch_size)
        if batch_norm:
            self.cat_conv = nn.Sequential(
                nn.Conv2d(c_up * 2, c_up, 1),
                nn.BatchNorm2d(c_up),
                nn.ReLU(True))
        else:
            self.cat_conv = nn.Sequential(
                nn.Conv2d(c_up * 2, c_up, 1),
                nn.ReLU(True))

    def forward(self, x_low, x_up):
        x_low_up = self.up(F.interpolate(
            x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x_attn = self.transformer(x_up, x_low_up)
        return self.cat_conv(torch.cat([x_up, x_attn], 1))


# =============================================================================
# 2.  DECODER NETWORK — verbatim from notebook Cell "vertenet-model"
#     heads = {'hm': NUM_CLASSES=1, 'reg': 2*NUM_CLASSES=2, 'wh': 2*4=8}
# =============================================================================

class DecNet_vertenet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super().__init__()
        self.dec_c2 = CombinationModule_proposed(64,  48,  batch_norm=True, patch_size=8)
        self.dec_c3 = CombinationModule_proposed(160, 64,  batch_norm=True, patch_size=8)
        self.dec_c4 = CombinationModule_proposed(256, 160, batch_norm=True, patch_size=8)
        self.heads  = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(
                    nn.Conv2d(channel, head_conv, 7, padding=3, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(head_conv, classes, 7, padding=3, bias=True))
            else:
                fc = nn.Sequential(
                    nn.Conv2d(channel, head_conv, 3, padding=1, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(head_conv, classes, final_kernel,
                              padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.__setattr__(head, fc)

    def forward(self, x):
        c4_combine = self.dec_c4(x[-1],  x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict


# =============================================================================
# 3.  TOP-LEVEL MODEL — verbatim from notebook Cell "vertenet-model"
#     CRITICAL attribute names matching checkpoint keys:
#       self.base_network   self.dec_net
#     heads used in training: {'hm':1, 'reg':2, 'wh':8}
# =============================================================================

class VerteNet(nn.Module):
    def __init__(self, heads, pretrained=False, down_ratio=4,
                 final_kernel=1, head_conv=256):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1       = int(np.log2(down_ratio))
        channels      = [3, 24, 48, 64, 160, 256]
        self.base_network = timm.create_model(
            'tf_efficientnetv2_s', pretrained=pretrained, features_only=True)
        self.dec_net = DecNet_vertenet(
            heads, final_kernel, head_conv, channels[self.l1])

    def forward(self, x):
        x1    = x
        feats = self.base_network(x)
        feats.insert(0, x1)           # matches notebook: feats.insert(0, x1)
        return self.dec_net(feats)


# =============================================================================
# 4.  DECODER — verbatim from notebook Cell "dec-decoder" / "best-model-viz"
# =============================================================================

class DecDecoder:
    def __init__(self, K=6, conf_thresh=0.1):
        self.K           = K
        self.conf_thresh = conf_thresh

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask].view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        b, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        return self._gather_feat(feat, ind)

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1,
                            padding=(kernel - 1) // 2)
        return heat * (hmax == heat).float()

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)
        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds // width).float()
        topk_xs   = (topk_inds %  width).float()
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys   = self._gather_feat(
            topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs   = self._gather_feat(
            topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)
        return topk_score, topk_inds, topk_ys, topk_xs

    def ctdet_decode(self, heat, wh, reg):
        """Global top-K decode. Returns numpy (K, 11)."""
        heat   = self._nms(heat)
        scores, inds, ys, xs = self._topk(heat)
        batch  = heat.size(0)

        scores = scores.view(batch, self.K, 1)
        reg    = self._tranpose_and_gather_feat(reg, inds).view(batch, self.K, 2)
        xs     = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys     = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh_out = self._tranpose_and_gather_feat(wh, inds).view(batch, self.K, 8)

        tl_x = xs - wh_out[:, :, 0:1];  tl_y = ys - wh_out[:, :, 1:2]
        tr_x = xs - wh_out[:, :, 2:3];  tr_y = ys - wh_out[:, :, 3:4]
        bl_x = xs - wh_out[:, :, 4:5];  bl_y = ys - wh_out[:, :, 5:6]
        br_x = xs - wh_out[:, :, 6:7];  br_y = ys - wh_out[:, :, 7:8]

        pts = torch.cat(
            [xs, ys, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, scores],
            dim=2).squeeze(0)
        return pts.data.cpu().numpy()

    def strip_decode(self, heat, wh, reg, n_strips=None):
        """Strip-based decode — one detection forced per vertical band."""
        n_strips = n_strips or self.K
        batch, c, height, width = heat.size()
        heat_nms = self._nms(heat)
        hm_np  = heat_nms[0, 0].cpu().numpy()
        wh_np  = wh[0].permute(1, 2, 0).cpu().numpy()
        reg_np = reg[0].permute(1, 2, 0).cpu().numpy()

        band_h = max(1, height // n_strips)
        rows   = []
        for s in range(n_strips):
            y0   = s * band_h
            y1   = height if s == n_strips - 1 else (s + 1) * band_h
            band = hm_np[y0:y1, :]
            if band.size == 0:
                continue
            flat_idx  = int(np.argmax(band))
            by, bx    = divmod(flat_idx, width)
            gy, gx    = y0 + by, bx
            score     = float(hm_np[gy, gx])
            rx, ry    = reg_np[gy, gx]
            cx, cy    = gx + rx, gy + ry
            w8        = wh_np[gy, gx]
            rows.append([cx, cy,
                         cx - w8[0], cy - w8[1],
                         cx - w8[2], cy - w8[3],
                         cx - w8[4], cy - w8[5],
                         cx - w8[6], cy - w8[7],
                         score])
        return np.array(rows, dtype=np.float32)


# =============================================================================
# 5.  COBB ANGLE CALCULATION — verbatim from notebook cobb_evaluate cell
# =============================================================================

def _is_S(mid_p_v):
    num = mid_p_v.shape[0]
    ll  = [(mid_p_v[i, 1] - mid_p_v[num-1, 1]) /
           (mid_p_v[0, 1] - mid_p_v[num-1, 1] + 1e-8) -
           (mid_p_v[i, 0] - mid_p_v[num-1, 0]) /
           (mid_p_v[0, 0] - mid_p_v[num-1, 0] + 1e-8)
           for i in range(num - 2)]
    ll = np.asarray(ll, np.float32)[:, None]
    lp = np.matmul(ll, ll.T)
    return abs(lp.sum() - np.abs(lp).sum()) > 1e-4


def cobb_angle_calc(pts):
    pts     = np.asarray(pts, np.float32)
    num_pts = pts.shape[0]
    vnum    = num_pts // 4 - 1

    mid_p_v = (pts[0::2] + pts[1::2]) / 2

    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i]     + pts[i + 2]) / 2
        pt2 = (pts[i + 1] + pts[i + 3]) / 2
        mid_p.extend([pt1, pt2])
    mid_p = np.asarray(mid_p, np.float32)

    vec_m   = mid_p[1::2] - mid_p[0::2]
    dot_v   = np.matmul(vec_m, vec_m.T)
    mod_v   = np.sqrt(np.sum(vec_m ** 2, axis=1))[:, None]
    mod_v   = np.matmul(mod_v, mod_v.T)
    cosines = np.clip(dot_v / (mod_v + 1e-8), 0., 1.)
    angles  = np.arccos(cosines)

    pos1  = np.argmax(angles, axis=1)
    maxt  = np.amax(angles, axis=1)
    pos2  = np.argmax(maxt)
    cobb1 = float(np.amax(maxt) / np.pi * 180)

    flag_s = _is_S(mid_p_v)
    if not flag_s:
        cobb2 = float(angles[0,    pos2]       / np.pi * 180)
        cobb3 = float(angles[vnum, pos1[pos2]] / np.pi * 180)
    else:
        angle2 = angles[pos2, :(pos2 + 1)]
        cobb2  = float(np.max(angle2) / np.pi * 180)
        angle3 = angles[pos1[pos2], pos1[pos2]:(vnum + 1)]
        cobb3  = float(np.max(angle3) / np.pi * 180)

    return [cobb1, cobb2, cobb3]   # [PT, MT, TL]


def interpret_cobb(angle_deg):
    if angle_deg < 10:   return "Normal (< 10°)"
    elif angle_deg < 25: return "Mild scoliosis (10°–25°)"
    elif angle_deg < 40: return "Moderate scoliosis (25°–40°)"
    else:                return "Severe scoliosis (> 40°) — surgical evaluation advised"


# =============================================================================
# 6.  PRE/POST-PROCESSING
# =============================================================================

def preprocess(image_bgr, input_h=512, input_w=512):
    img = cv2.resize(image_bgr, (input_w, input_h))
    img = img.astype(np.float32) / 255.0 - 0.5
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def decode_detections(dets_np, orig_h, orig_w, input_h, input_w,
                      down_ratio, conf_thresh=0.0):
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    def hm2orig(x_hm, y_hm):
        return float(x_hm) * down_ratio * scale_x, float(y_hm) * down_ratio * scale_y

    results = []
    for row in dets_np:
        score = float(row[10])
        if score < conf_thresh:
            continue
        cx, cy      = hm2orig(row[0], row[1])
        tl_x, tl_y = hm2orig(row[2], row[3])
        tr_x, tr_y = hm2orig(row[4], row[5])
        bl_x, bl_y = hm2orig(row[6], row[7])
        br_x, br_y = hm2orig(row[8], row[9])
        results.append({
            'center_x':      cx,
            'center_y':      cy,
            'score':         score,
            'corners':       np.array([[tl_x, tl_y], [tr_x, tr_y],
                                        [bl_x, bl_y], [br_x, br_y]], np.float32),
            'corner_labels': ['TL', 'TR', 'BL', 'BR'],
        })
    results.sort(key=lambda d: d['center_y'])
    return results


def build_pts_for_cobb(vertebrae):
    all_pts = []
    for v in vertebrae:
        all_pts.extend(v['corners'].tolist())
    return np.array(all_pts, np.float32)


# =============================================================================
# 7.  WEIGHT LOADING
# =============================================================================

def load_weights(model, weights_path, device='cpu'):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict):
        sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        sd = ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(sd.keys())
    matched    = model_keys & ckpt_keys

    if matched:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(model_keys) - len(missing)
        print(f"  ✅ {n_loaded}/{len(model_keys)} tensors loaded  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if len(missing) > 0:
            print(f"     Missing keys  : {list(missing)[:5]}")
            print(f"     Unexpected    : {list(unexpected)[:5]}")
    else:
        print("  ❌ 0 matching keys — weights NOT loaded.")
    return model


# =============================================================================
# 8.  VISUALISATION
# =============================================================================

_SPINE_COLORS = [
    (0,   200, 255), (100, 255, 100), (255, 180,  50),
    (255,  80, 180), (180,  80, 255), ( 50, 180, 255),
]
_CORNER_COLORS = {
    'TL': (0, 200, 255), 'TR': (255, 200, 0),
    'BL': (0, 255, 100), 'BR': (255,  80, 80),
}


def draw_vertebrae(img_bgr, vertebrae):
    out = img_bgr.copy()
    for vi, v in enumerate(vertebrae):
        col     = _SPINE_COLORS[vi % len(_SPINE_COLORS)]
        corners = v['corners'].astype(int)
        order   = [0, 1, 3, 2]
        for j in range(4):
            cv2.line(out, tuple(corners[order[j]]),
                     tuple(corners[order[(j+1) % 4]]), col, 2, cv2.LINE_AA)
        for ci, label in enumerate(v['corner_labels']):
            cv2.circle(out, tuple(corners[ci]), 5, _CORNER_COLORS[label], -1, cv2.LINE_AA)
            cv2.circle(out, tuple(corners[ci]), 6, (0, 0, 0), 1, cv2.LINE_AA)
        cx, cy = int(v['center_x']), int(v['center_y'])
        cv2.drawMarker(out, (cx, cy), col, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        txt = f"V{vi+1}  {v['score']:.3f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx+10, cy-th-4), (cx+10+tw, cy+4), (0, 0, 0), -1)
        cv2.putText(out, txt, (cx+10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_cobb_lines(img_bgr, vertebrae, cobb_angles):
    out = img_bgr.copy()
    if len(vertebrae) < 2:
        return out

    def _endplate(corners, col):
        c = corners.astype(float)
        for a, b in [(0, 1), (2, 3)]:
            mid  = (c[a] + c[b]) / 2
            dirv = c[b] - c[a]
            norm = np.linalg.norm(dirv)
            if norm < 1:
                continue
            dirv /= norm
            cv2.line(out, tuple((mid - dirv*50).astype(int)),
                     tuple((mid + dirv*50).astype(int)), col, 2, cv2.LINE_AA)

    for vi, v in enumerate(vertebrae):
        _endplate(v['corners'], _SPINE_COLORS[vi % len(_SPINE_COLORS)])

    h, w   = out.shape[:2]
    bx, by = max(w - 270, 0), 16
    for i, (txt, col) in enumerate([
        (f"PT (Primary) : {cobb_angles[0]:.1f}\u00b0", (0, 255, 255)),
        (f"MT (Main Th.): {cobb_angles[1]:.1f}\u00b0", (255, 200, 50)),
        (f"TL (Thor/Lum): {cobb_angles[2]:.1f}\u00b0", (100, 255, 100)),
    ]):
        y = by + i * 30
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (bx-6, y-th-5), (bx+tw+6, y+5), (0, 0, 0), -1)
        cv2.putText(out, txt, (bx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    flag_txt = "SCOLIOSIS DETECTED" if cobb_angles[0] >= 10 else "Normal (< 10\u00b0)"
    flag_col = (0, 60, 255) if cobb_angles[0] >= 10 else (50, 220, 50)
    cv2.putText(out, flag_txt, (bx-6, by + 3*30 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, flag_col, 2, cv2.LINE_AA)
    return out


def draw_heatmap_overlay(img_bgr, hm_np):
    hm = hm_np[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color   = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_resized = cv2.resize(hm_color, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 0.55, hm_resized, 0.45, 0)


# =============================================================================
# 9.  HIGH-LEVEL INFERENCE CLASS
# =============================================================================

class VerteNetInference:
    """
    Main class for groupmates.

    Usage:
        model  = VerteNetInference(weights_path="exp1_baseline_phase1_best.pth")
        result = model.visualize("spine.jpg", out_dir="./output")
        print(result['cobb_angles'])   # {'PT': float, 'MT': float, 'TL': float}
    """

    # Exact heads used during training in the notebook
    HEADS = {'hm': 1, 'reg': 2, 'wh': 8}   # NUM_CLASSES=1, reg=2*1, wh=2*4

    def __init__(self, weights_path,
                 head_conv=512, down_ratio=4, input_size=512,
                 topk=17, conf_thresh=0.05, mode='strip', device=None):
        self.device      = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.down_ratio  = down_ratio
        self.input_size  = input_size
        self.conf_thresh = conf_thresh
        self.topk        = topk
        self.mode        = mode

        print(f"\n[VerteNet] Initialising on {self.device}")
        self.model = VerteNet(
            heads=self.HEADS, pretrained=False,
            down_ratio=down_ratio, final_kernel=1, head_conv=head_conv,
        ).to(self.device)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        print(f"[VerteNet] Loading weights : {weights_path}")
        self.model = load_weights(self.model, weights_path, device=str(self.device))
        self.model.eval()

        n = sum(p.numel() for p in self.model.parameters())
        print(f"[VerteNet] Parameters : {n:,}  (~{n/1e6:.2f} M)\n")
        self.decoder = DecDecoder(K=topk, conf_thresh=conf_thresh)

    def predict(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot open image: {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        tensor = preprocess(img_bgr, self.input_size, self.input_size).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        hm  = outputs['hm'].cpu()
        reg = outputs['reg'].cpu()
        wh  = outputs['wh'].cpu()

        if self.mode == 'strip':
            dets_np   = self.decoder.strip_decode(hm, wh, reg, n_strips=self.topk)
            vertebrae = decode_detections(
                dets_np, orig_h, orig_w, self.input_size, self.input_size,
                self.down_ratio, conf_thresh=0.0)
        else:
            dets_np   = self.decoder.ctdet_decode(hm, wh, reg)
            vertebrae = decode_detections(
                dets_np, orig_h, orig_w, self.input_size, self.input_size,
                self.down_ratio, self.conf_thresh)

        cobb = [0.0, 0.0, 0.0]
        if len(vertebrae) >= 2:
            try:
                cobb = cobb_angle_calc(build_pts_for_cobb(vertebrae))
            except Exception as e:
                print(f"  [WARN] Cobb failed: {e}")

        return {
            'image_path'    : image_path,
            'n_detected'    : len(vertebrae),
            'vertebrae'     : [
                {'index': i+1, 'score': v['score'],
                 'center_x': v['center_x'], 'center_y': v['center_y'],
                 'corners': {'TL': v['corners'][0].tolist(),
                             'TR': v['corners'][1].tolist(),
                             'BL': v['corners'][2].tolist(),
                             'BR': v['corners'][3].tolist()}}
                for i, v in enumerate(vertebrae)],
            'cobb_angles'   : {'PT': cobb[0], 'MT': cobb[1], 'TL': cobb[2]},
            'scoliosis'     : cobb[0] >= 10,
            'interpretation': interpret_cobb(cobb[0]),
            'heatmap_stats' : {'min': float(hm.min()),
                               'max': float(hm.max()),
                               'mean': float(hm.mean())},
            '_raw'          : {'hm': hm.numpy(), 'vertebrae_list': vertebrae},
        }

    def visualize(self, image_path, out_dir='./output', save_json=True):
        os.makedirs(out_dir, exist_ok=True)
        result    = self.predict(image_path)
        img_bgr   = cv2.imread(image_path)
        vertebrae = result['_raw']['vertebrae_list']
        hm_np     = result['_raw']['hm']
        cobb      = [result['cobb_angles']['PT'],
                     result['cobb_angles']['MT'],
                     result['cobb_angles']['TL']]
        n_det     = result['n_detected']
        base      = os.path.splitext(os.path.basename(image_path))[0]

        img_ann  = draw_vertebrae(img_bgr, vertebrae)
        img_cobb = draw_cobb_lines(img_ann, vertebrae, cobb)
        cv2.imwrite(os.path.join(out_dir, f"annotated_{base}.png"), img_cobb)

        hm_img = draw_heatmap_overlay(img_bgr, hm_np)
        cv2.imwrite(os.path.join(out_dir, f"heatmap_{base}.png"), hm_img)

        fig, axes = plt.subplots(1, 3, figsize=(21, 8))
        axes[0].imshow(cv2.cvtColor(img_bgr,   cv2.COLOR_BGR2RGB)); axes[0].axis('off')
        axes[0].set_title('Original Image', fontsize=13)
        axes[1].imshow(cv2.cvtColor(img_cobb,  cv2.COLOR_BGR2RGB)); axes[1].axis('off')
        axes[1].set_title(
            f"Vertebrae + Cobb  (n={n_det})\n"
            f"PT={cobb[0]:.1f}\u00b0  MT={cobb[1]:.1f}\u00b0  TL={cobb[2]:.1f}\u00b0\n"
            f"{result['interpretation']}", fontsize=10)
        axes[2].imshow(cv2.cvtColor(hm_img,    cv2.COLOR_BGR2RGB)); axes[2].axis('off')
        axes[2].set_title('Heatmap Overlay', fontsize=13)
        if vertebrae:
            ax_in = fig.add_axes([0.685, 0.07, 0.29, 0.20])
            scores  = [v['score'] for v in vertebrae]
            xlabels = [f"V{i+1}" for i in range(len(vertebrae))]
            ax_in.bar(xlabels, scores, color=[
                f"#{_SPINE_COLORS[i%6][2]:02x}{_SPINE_COLORS[i%6][1]:02x}"
                f"{_SPINE_COLORS[i%6][0]:02x}" for i in range(len(vertebrae))])
            ax_in.set_ylim(0, 1)
            ax_in.axhline(self.conf_thresh, color='red', ls='--', lw=1,
                          label=f'thresh={self.conf_thresh}')
            ax_in.set_title('Confidence Scores', fontsize=8)
            ax_in.tick_params(labelsize=7); ax_in.legend(fontsize=6)
        plt.suptitle(f"VerteNet — {os.path.basename(image_path)}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overview_{base}.png"),
                    bbox_inches='tight', dpi=130)
        plt.close(fig)

        print(f"\n[VerteNet] Results for {os.path.basename(image_path)}")
        print(f"  Vertebrae detected : {n_det}")
        for i, v in enumerate(vertebrae):
            c = v['corners']
            print(f"  V{i+1:2d}  centre=({v['center_x']:6.1f},{v['center_y']:6.1f})  "
                  f"score={v['score']:.4f}  "
                  f"TL=({c[0][0]:.0f},{c[0][1]:.0f})  TR=({c[1][0]:.0f},{c[1][1]:.0f})")
        print(f"  PT (Primary) Cobb  : {cobb[0]:.2f}\u00b0")
        print(f"  MT (Main Th.)      : {cobb[1]:.2f}\u00b0")
        print(f"  TL (Thor/Lum)      : {cobb[2]:.2f}\u00b0")
        print(f"  \u2192 {result['interpretation']}")

        if save_json:
            serial = {k: v for k, v in result.items() if k != '_raw'}
            with open(os.path.join(out_dir, f"result_{base}.json"), 'w') as f:
                json.dump(serial, f, indent=2)
        return result

    def predict_batch(self, image_paths, out_dir='./output', save_json=True):
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] {path}")
            try:
                results.append(self.visualize(path, out_dir=out_dir, save_json=save_json))
            except Exception as e:
                print(f"  [ERROR] {e}")
        return results
