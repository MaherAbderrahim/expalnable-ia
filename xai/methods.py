"""XAI methods for the 3D MONAI UNet ensemble (PI-CAI csPCa).

Implemented methods
-------------------
1. Grad-CAM / Grad-CAM++ (3D): the PI-CAI UNet has no full-resolution
   pre-classification feature map (its top up-path maps 64 ch -> 2 ch at full
   res, i.e. straight into the logits). The CAM feature block is therefore the
   LAST skip-connection of the network (model.model[1], the SkipConnection of
   the top block in this "enhanced UNet": nn.Sequential(down, SkipConnection,
   up-path)): its output concatenates the top down-path features with the
   deeper decoder features (96 ch = 32+64 for the released net, at half
   resolution) — the last multi-channel semantic feature map before the
   2-channel classification path. Grad-CAM weights = GAP of the class
   gradient; Grad-CAM++ = the second-order formulation of Chattopadhyay et
   al. 2018. CAM computed per fold then averaged over the 5 folds (matching
   how the ensemble itself averages), trilinearly resized to the input grid,
   ReLU + min-max normalized.

2. Integrated Gradients: target f = csPCa logit summed over the volume,
   baseline = zeros (= per-voxel channel mean of the z-scored input),
   mean over the 5 folds, 32 alpha steps, summarized over input channels by
   max of |IG| along the channel axis.

3. Occlusion sensitivity: occlude rectangular slabs of the input (fill with 0
   = the per-voxel mean of the z-scored channels) and measure the drop of the
   model's fused csPCa score:  score(x) = max over the model grid of the
   fold-averaged raw ensemble heatmap (sigma=0 proxy of the official fusion,
   chosen for speed; documented as a proxy).

All functions operate on a SINGLE preprocessed tensor (1, 3, 20, 256, 256)
and explain the SAME relevant output: the csPCa channel (channel index 1).
Invalid inputs are rejected upfront by `_validate_input` (ValueError).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

_OCCLUSION_BATCH = 32

EXPECTED_IN_CHANNELS = 3


def _validate_input(input_tensor: torch.Tensor, single_case: bool = True) -> torch.Tensor:
    """Validate the input contract shared by the XAI methods.

    single_case=True (attribution functions) requires exactly B=1;
    single_case=False (batch scorers such as fused_score_from_batch) allows
    any batch size. Raises ValueError (never silently reshapes/moves) so
    that shapes stay honest between preprocessing and the explanation.
    """
    if input_tensor.dim() != 5:
        raise ValueError(
            f"expected a 5-D tensor (B, C, d, h, w), got {tuple(input_tensor.shape)}"
        )
    if single_case and input_tensor.shape[0] != 1:
        raise ValueError(
            "XAI methods explain ONE case at a time (B=1); "
            f"got batch of {input_tensor.shape[0]}"
        )
    if input_tensor.shape[1] != EXPECTED_IN_CHANNELS:
        raise ValueError(
            f"the PI-CAI model expects C={EXPECTED_IN_CHANNELS} channels (t2w, adc, hbv); "
            f"got C={input_tensor.shape[1]}"
        )
    return input_tensor


# ---------------------------------------------------------------------------
# Grad-CAM / Grad-CAM++ (3D)
# ---------------------------------------------------------------------------

def _top_block_last_skip(model: torch.nn.Module) -> torch.nn.Module:
    """The 3-module top block is Sequential(down, SkipConnection, up); the
    middle module is the LAST skip connection whose output the CAM reads."""
    return model.model[1]


class _FeatureHook:
    """Stores the (graph-connected) output feature block of the hooked module."""

    def __init__(self, module: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.activations = output

    def remove(self):
        self.handle.remove()


def cam_weights(grads: torch.Tensor, method: str) -> torch.Tensor:
    """Per-channel CAM weights (B, C) from gradient maps (B, C, d, h, w)."""
    if method == "gradcam":
        return grads.mean(dim=(2, 3, 4))
    raise ValueError(method)


def cam_weights_gradcampp(grads: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++ per-voxel formulation (Chattopadhyay et al. 2018, Eq. 13-15):
    alpha_ki = g2_ki / (2*g2_ki + sum_j A_kj*g2_kj + eps), channel weight =
    sum_i alpha_ki * g_ki. (The per-channel S_k=sum_j A*g2 makes alpha
    self-normalizing; direct 1/(2*sum g2 + ...) variants are numerically
    unstable and produced page-corner artifacts on this model.)"""
    g2 = grads ** 2
    eps = 1e-8
    S = (g2 * activations).sum(dim=(2, 3, 4), keepdim=True)             # (B, C, 1, 1, 1)
    alpha = g2 / (2.0 * g2 + S + eps)                                   # (B, C, d, h, w)
    return (alpha * grads).sum(dim=(2, 3, 4))                           # (B, C)


def gradcam_3d(model: torch.nn.Module, input_tensor: torch.Tensor,
               method: str = "gradcam") -> np.ndarray:
    """CAM of ONE fold model: (20, 256, 256) in [0, 1]."""
    _validate_input(input_tensor)
    hook_module = _top_block_last_skip(model)
    hook = _FeatureHook(hook_module)
    try:
        inp = input_tensor.clone()
        out = model(inp)
        score = out[:, 1].sum()
        grads, = torch.autograd.grad(score, hook.activations, retain_graph=False)
        A = hook.activations
        if method == "gradcam":
            weights = cam_weights(grads, "gradcam")                       # (B, C)
            cam = torch.einsum("bc,bcdhw->bdhw", weights, A)
        else:
            weights = cam_weights_gradcampp(grads, A)
            cam = torch.einsum("bc,bcdhw->bdhw", weights, A)
        cam = torch.relu(cam)[0].detach()
    finally:
        hook.remove()
    if cam.shape[-3:] != input_tensor.shape[-3:]:
        cam = F.interpolate(
            cam[None, None].float(), size=input_tensor.shape[-3:],
            mode="trilinear", align_corners=False,
        )[0, 0]
    cam = cam.cpu().numpy()
    cam = cam.astype(np.float32)
    cam -= cam.min()
    cam /= (cam.max() + 1e-12)
    return cam


def ensemble_gradcam(models, input_tensor: torch.Tensor, device: str,
                     method: str = "gradcam") -> np.ndarray:
    _validate_input(input_tensor)
    cams = [gradcam_3d(m, input_tensor.to(next(m.parameters()).device), method) for m in models]
    cam = np.mean(cams, axis=0).astype(np.float32)
    cam -= cam.min()
    cam /= (cam.max() + 1e-12)
    return cam


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients_single(model: torch.nn.Module, input_tensor: torch.Tensor,
                                device: str, steps: int = 32) -> np.ndarray:
    """IG of ONE fold model: (3, d, h, w) RAW signed attribution, not normalized.

    Signed output preserved here: negative attributions (voxels that, along
    the path, push the csPCa score DOWN) keep their sign, which is meaningful
    for gradient-path methods. Callers choose their own summarization.
    """
    _validate_input(input_tensor)
    model = model.to(device)
    x0 = input_tensor.to(device, dtype=torch.float32).clone().requires_grad_(True)
    base = torch.zeros_like(x0, device=device)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]
    acc = torch.zeros_like(x0, device=device)
    for a in alphas:
        x = base + (x0 - base) * a
        out = model(x)
        score = out[:, 1].sum()
        grads, = torch.autograd.grad(score, x)
        acc += grads.detach()
    return ((x0 - base) * acc / steps).detach().cpu().numpy()[0]


def ensemble_integrated_gradients(models, input_tensor: torch.Tensor, device: str,
                                  steps: int = 32) -> np.ndarray:
    """(20, 256, 256) attribution map in [0, 1], averaged over fold models.

    Fold-average the RAW signed IG per channel, then summarize over input
    channels by max of |IG| along the channel axis (positives and negatives
    are collapsed to an intensity; the sign of the mean is NOT displayed —
    documented limitation). Min-max normalized to [0, 1].
    """
    _validate_input(input_tensor)
    total = torch.zeros(input_tensor.shape, dtype=torch.float32, device=device)
    for m in models:
        m = m.to(device)
        ig_raw = integrated_gradients_single(m, input_tensor, device, steps=steps)
        total += torch.from_numpy(ig_raw).to(device)
    ig = total.detach().cpu().numpy()[0]          # (3, d, h, w)
    ig = np.abs(ig).max(axis=0)                   # summarize over channels
    ig = ig.astype(np.float32)
    ig -= ig.min()
    ig /= (ig.max() + 1e-12)
    return ig


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------

def fused_score_from_batch(models, xb: torch.Tensor, device: str) -> np.ndarray:
    """max over the model grid of the fold-averaged sigmoid maps (no sigma, no
    TTA: occlusion-speed proxy documented in the module docstring)."""
    _validate_input(xb, single_case=False)
    acc = None
    for m in models:
        p = torch.sigmoid(m(xb.to(device, dtype=torch.float32)))[:, 1, ...].detach().cpu().numpy()
        acc = p if acc is None else acc + p
    fused = acc / len(models)
    return fused.reshape(len(xb), -1).max(axis=1)


def occlusion_scan(models, input_tensor: torch.Tensor, device: str,
                   slab=(2, 32, 32), stride=(2, 24, 24),
                   normalize: bool = True) -> np.ndarray:
    """Per-slab score drop: (z, y, x) array in [0, 1] (great = model lost score).

    Occluder = rectangular slab filled with 0.0 (the per-voxel mean of the
    z-scored channels). Overlapping slabs: the LAST placed drop wins (write
    order = scan order, documented). The score whose drop is measured is the
    fold-averaged fused max-sigmoid proxy (see fused_score_from_batch).

    normalize=True (default, historical behavior): min-max scaled to [0, 1] —
    magnitude information is discarded. normalize=False returns the RAW drop
    magnitudes (score units, clipped at 0) for the scientific controls (input
    independence, flat-input/"nothing found" checks); raw drops below 1e-9
    scale are not artificially rescaled.
    """
    _validate_input(input_tensor)
    x0 = input_tensor.to(device, dtype=torch.float32)
    d, h, w = x0.shape[-3:]
    base = fused_score_from_batch(models, x0, device)[0]
    positions = [
        (zi, yi, xi)
        for zi in range(0, max(d - slab[0], 0) + 1, stride[0])
        for yi in range(0, max(h - slab[1], 0) + 1, stride[1])
        for xi in range(0, max(w - slab[2], 0) + 1, stride[2])
    ]
    out = np.zeros((d, h, w), dtype=np.float32)
    for chunk_start in range(0, len(positions), _OCCLUSION_BATCH):
        chunk = positions[chunk_start: chunk_start + _OCCLUSION_BATCH]
        xb = x0.repeat(len(chunk), *([1] * (x0.dim() - 1)))
        for bi, (zi, yi, xi) in enumerate(chunk):
            xb[bi, :, zi: zi + slab[0], yi: yi + slab[1], xi: xi + slab[2]] = 0.0
        scores = fused_score_from_batch(models, xb, device)
        drop = np.clip(base - scores, 0.0, None).astype(np.float32)
        for (zi, yi, xi), dp in zip(chunk, drop):
            out[zi: zi + slab[0], yi: yi + slab[1], xi: xi + slab[2]] = dp
    mx = out.max()
    if normalize and mx > 0:
        out /= mx
    return out
