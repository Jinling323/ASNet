"""SHA jpg/point-npy adapter. Density maps always preserve annotation counts."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


def point_density(points, height, width, kernel_size=15, sigma=4.0):
    """Stamp normalized Gaussians, including at image boundaries and duplicate points."""
    density = np.zeros((height, width), dtype=np.float32)
    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma ** 2))
    for x, y in points[:, :2]:
        # A few SHA annotations slightly exceed the boundary after resizing.
        x = int(np.clip(np.floor(x), 0, width - 1))
        y = int(np.clip(np.floor(y), 0, height - 1))
        left, right = max(0, x - radius), min(width, x + radius + 1)
        top, bottom = max(0, y - radius), min(height, y + radius + 1)
        patch = kernel[top - y + radius:bottom - y + radius,
                       left - x + radius:right - x + radius]
        density[top:bottom, left:right] += (patch / patch.sum()).astype(np.float32)
    return density


def pooled_density(density):
    """Sliding 8x8 mean on the network's 1/8-resolution density map."""
    return F.avg_pool2d(F.pad(density, (4, 3, 4, 3)), 8, stride=1)


def attention_labels(density, threshold):
    pooled = pooled_density(density)
    return ((pooled > 0).long() + (pooled > threshold).long()).squeeze(1)


class SHADataset(Dataset):
    def __init__(self, directory, crop_size=None, patches_per_image=1,
                 kernel_size=15, sigma=4.0, color_jitter=0.2, max_size=0):
        self.directory = Path(directory)
        self.images = sorted(p for p in self.directory.iterdir()
                             if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.images:
            raise ValueError('No images found in {}'.format(self.directory))
        missing = [str(p.with_suffix('.npy')) for p in self.images
                   if not p.with_suffix('.npy').is_file()]
        if missing:
            raise FileNotFoundError('Missing point annotations: {}'.format(missing[:5]))
        self.crop_size = crop_size
        self.patches_per_image = patches_per_image if crop_size else 1
        self.kernel_size, self.sigma = kernel_size, sigma
        self.color_jitter, self.max_size = color_jitter, max_size

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def load_image(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        points = np.load(path.with_suffix('.npy'), allow_pickle=False)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError('{}: expected Nx2 or Nx3 point annotations'.format(path))
        points = points[:, :2].astype(np.float64, copy=True)
        if not np.isfinite(points).all():
            raise ValueError('{}: non-finite coordinates'.format(path))
        if self.max_size and max(image.size) > self.max_size:
            width, height = image.size
            ratio = self.max_size / max(width, height)
            new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
            points *= np.array(new_size) / np.array(image.size)
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        return path, image, points

    def __getitem__(self, index):
        path, image, points = self.load_image(index // self.patches_per_image)
        width, height = image.size
        density = point_density(points, height, width, self.kernel_size, self.sigma)
        roi = np.ones((height, width), dtype=np.float32)
        if self.crop_size:
            size = self.crop_size
            pad_w, pad_h = max(0, size - width), max(0, size - height)
        else:
            pad_w, pad_h = (-width) % 16, (-height) % 16
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h))
        density = np.pad(density, ((0, pad_h), (0, pad_w)))
        roi = np.pad(roi, ((0, pad_h), (0, pad_w)))

        if self.crop_size:
            left = random.randint(0, image.width - size)
            top = random.randint(0, image.height - size)
            image = image.crop((left, top, left + size, top + size))
            density = density[top:top + size, left:left + size]
            roi = roi[top:top + size, left:left + size]
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                density, roi = density[:, ::-1], roi[:, ::-1]
            if self.color_jitter:
                transforms = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
                random.shuffle(transforms)
                for transform in transforms:
                    image = transform(image).enhance(random.uniform(1 - self.color_jitter,
                                                                   1 + self.color_jitter))

        image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        density = torch.from_numpy(np.ascontiguousarray(density))[None, None]
        density = F.avg_pool2d(density, 8, stride=8)[0] * 64
        return {'image': image, 'density': density,
                'roi': torch.from_numpy(np.ascontiguousarray(roi))[None],
                'count': float(len(points)), 'name': path.stem}


def estimate_thresholds(dataset, sample_count=64, seed=42):
    """Estimate attention and APLoss thresholds using training images only."""
    indices = np.random.default_rng(seed).choice(len(dataset.images),
                                               min(sample_count, len(dataset.images)), replace=False)
    attention_sum = attention_n = pyramid_sum = pyramid_n = 0
    for index in indices:
        _, image, points = dataset.load_image(int(index))
        density = point_density(points, image.height, image.width,
                                dataset.kernel_size, dataset.sigma)
        # Multiples of 64 align input /8 and both levels of APLoss.
        density = np.pad(density, ((0, (-image.height) % 64), (0, (-image.width) % 64)))
        density = torch.from_numpy(density)[None, None]
        density = F.avg_pool2d(density, 8, stride=8) * 64
        for values, kind in [(pooled_density(density), 'attention'),
                             (F.avg_pool2d(density, 8, stride=8) * 64, 'pyramid')]:
            active = values[values > 0]
            if kind == 'attention':
                attention_sum += active.sum().item()
                attention_n += active.numel()
            else:
                pyramid_sum += active.sum().item()
                pyramid_n += active.numel()
    if not attention_n or not pyramid_n:
        raise ValueError('No annotated heads in threshold samples; increase --threshold-samples')
    return {'attention': attention_sum / attention_n, 'pyramid': pyramid_sum / pyramid_n}
