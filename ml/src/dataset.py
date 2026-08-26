"""
Patch-based dataset for training on a single large aerial tile.
Mirrors the tiling strategy real orthophoto pipelines use (SAHI-style
slicing) — necessary because full orthophotos are far too large to feed
into a CNN directly, and because it multiplies one labeled tile into many
training samples.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Patch dataset over a specified (y0,y1,x0,x1) region of the tile only.

    IMPORTANT: train and val datasets must be built over NON-OVERLAPPING
    spatial regions of the source tile (see split_regions below). Building
    them via a random split of overlapping sliding-window patches (as the
    first version of this pipeline did) leaks pixels between train and val,
    since neighboring patches share most of their area -> inflates the
    reported validation IoU. This version enforces a hard spatial boundary.
    """
    def __init__(self, rgb, mask, region, patch_size=128, stride=64, augment=True):
        self.rgb = rgb
        self.mask = mask
        self.patch_size = patch_size
        self.augment = augment

        y0, y1, x0, x1 = region
        self.coords = []
        for y in range(y0, y1 - patch_size + 1, stride):
            for x in range(x0, x1 - patch_size + 1, stride):
                self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p = self.patch_size
        img = self.rgb[y:y + p, x:x + p, :].copy()
        msk = self.mask[y:y + p, x:x + p].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()

            # Color jitter: brightness/contrast perturbation. Aerial imagery
            # varies with season/sun-angle/sensor across real drone passes;
            # a model trained on only one tile's exact color palette
            # overfits to lighting conditions that won't hold at other
            # villages. This is a cheap, high-value robustness augmentation.
            if np.random.rand() < 0.7:
                brightness = np.random.uniform(-0.08, 0.08)
                contrast = np.random.uniform(0.85, 1.15)
                img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)
            if np.random.rand() < 0.3:
                # slight per-channel gain to simulate sensor/white-balance drift
                gains = np.random.uniform(0.92, 1.08, size=3)
                img = np.clip(img * gains, 0, 1)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        msk_t = torch.from_numpy(msk).float().unsqueeze(0)
        return img_t, msk_t


class MultiScalePatchDataset(Dataset):
    """Samples patches at multiple physical sizes (e.g. 96/128/160/192px)
    and resizes each to a fixed network input size (128x128).

    Why this matters: a fixed-size sliding window means the model always
    sees buildings at whatever apparent size they happen to be at that one
    scale - a small house and a large industrial shed look like completely
    different "amounts of the patch" to the network. Training on randomly
    varied physical scales, all resampled to the same input resolution,
    forces the model to learn scale-invariant building features instead of
    memorizing "buildings are roughly this many pixels wide", which is
    exactly the kind of thing that breaks when the model sees a real
    SVAMITVA village with a different building-size distribution than this
    demo tile.
    """
    def __init__(self, rgb, mask, region, net_size=128, scales=(96, 128, 160, 192), stride_frac=0.4, augment=True):
        self.rgb = rgb
        self.mask = mask
        self.net_size = net_size
        self.scales = scales
        self.augment = augment

        y0, y1, x0, x1 = region
        # build a coordinate list per scale so __len__ combines all scales
        self.samples = []  # (scale, y, x)
        for s in scales:
            stride = max(8, int(s * stride_frac))
            for y in range(y0, y1 - s + 1, stride):
                for x in range(x0, x1 - s + 1, stride):
                    self.samples.append((s, y, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, y, x = self.samples[idx]
        img = self.rgb[y:y + s, x:x + s, :].copy()
        msk = self.mask[y:y + s, x:x + s].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy(); msk = np.fliplr(msk).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy(); msk = np.flipud(msk).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy(); msk = np.rot90(msk, k).copy()
            if np.random.rand() < 0.7:
                brightness = np.random.uniform(-0.08, 0.08)
                contrast = np.random.uniform(0.85, 1.15)
                img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)

        # resize (physical scale -> fixed network input) via cv2, area
        # interpolation for downsizing (anti-aliased), linear for upsizing
        net = self.net_size
        interp = cv2.INTER_AREA if s > net else cv2.INTER_LINEAR
        img_r = cv2.resize(img, (net, net), interpolation=interp)
        msk_r = cv2.resize(msk, (net, net), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_r.transpose(2, 0, 1)).float()
        msk_t = torch.from_numpy(msk_r).float().unsqueeze(0)
        return img_t, msk_t


def split_regions(H, W, val_fraction=0.2, axis='x', side='right'):
    """Hard spatial split: reserve a contiguous strip of the tile for
    validation that the training set never touches (not even overlapping
    patch borders), giving an honest held-out evaluation.
    axis='x' splits left/right, axis='y' splits top/bottom.
    side picks which edge becomes the held-out validation region -
    used for k-fold spatial cross-validation (rotate through all 4 edges
    to confirm the reported metric isn't a fluke of one particular split)."""
    if axis == 'x':
        split = int(W * (1 - val_fraction))
        if side == 'right':
            train_region = (0, H, 0, split)
            val_region = (0, H, split, W)
        else:  # left
            val_w = W - split
            train_region = (0, H, val_w, W)
            val_region = (0, H, 0, val_w)
    else:
        split = int(H * (1 - val_fraction))
        if side == 'bottom':
            train_region = (0, split, 0, W)
            val_region = (split, H, 0, W)
        else:  # top
            val_h = H - split
            train_region = (val_h, H, 0, W)
            val_region = (0, val_h, 0, W)
    return train_region, val_region
