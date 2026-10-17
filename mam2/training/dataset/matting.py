"""Image-matting data for the independent lightweight-matter training stage."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mam2.training.dataset.augmentations import (
    Composite,
    CutMask,
    GenMask,
    RandomAffine,
    RandomCrop,
    RandomJitter,
    ToTensor,
    maybe_random_interp,
)
from mam2.training.dataset.images import SUFFIXES, paired_images


class ImageMattingDataset(Dataset):
    def __init__(self, sources, background_folder, augmentation):
        self.pairs = paired_images(sources)
        self.backgrounds = sorted(
            str(p) for p in Path(background_folder).iterdir() if p.suffix.lower() in SUFFIXES
        )
        if not self.backgrounds:
            raise ValueError("At least one training background is required")
        self.config = augmentation
        random_interp = augmentation["random_interpolation"]
        self.transform = Compose(
            [
                RandomAffine(
                    degrees=augmentation["degrees"],
                    scale=augmentation["scale"],
                    shear=augmentation["shear"],
                    flip=augmentation["flip"],
                    random_interpolation=random_interp,
                ),
                GenMask(random_interpolation=random_interp),
                CutMask(perturb_prob=augmentation["cutmask_prob"]),
                RandomCrop((augmentation["crop_size"],) * 2, random_interpolation=random_interp),
                RandomJitter(),
                Composite(),
                ToTensor(phase="train"),
            ]
        )

    def __len__(self):
        return max(len(self.pairs), len(self.backgrounds))

    @staticmethod
    def _read(path, grayscale=False):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read training image: {path}")
        return image.astype(np.float32) / 255 if grayscale else image

    def __getitem__(self, index):
        foreground_path, alpha_path = self.pairs[index % len(self.pairs)]
        foreground = self._read(foreground_path)
        alpha = self._read(alpha_path, grayscale=True)
        if foreground.shape[:2] != alpha.shape:
            raise ValueError(f"Foreground/alpha dimensions differ: {foreground_path}")
        background = self._read(self.backgrounds[index % len(self.backgrounds)])
        random_interp = self.config["random_interpolation"]
        if np.random.random() < self.config["composite_prob"]:
            other_fg, other_alpha = self.pairs[np.random.randint(len(self.pairs))]
            other_fg = self._read(other_fg)
            other_alpha = self._read(other_alpha, grayscale=True)
            size = (alpha.shape[1], alpha.shape[0])
            other_fg = cv2.resize(
                other_fg, size, interpolation=maybe_random_interp(cv2.INTER_NEAREST, random_interp)
            )
            other_alpha = cv2.resize(
                other_alpha,
                size,
                interpolation=maybe_random_interp(cv2.INTER_NEAREST, random_interp),
            )
            combined = 1 - (1 - alpha) * (1 - other_alpha)
            if np.any(combined < 1):
                foreground = (
                    foreground * alpha[..., None] + other_fg * (1 - alpha[..., None])
                ).astype(np.uint8)
                alpha = combined
        if np.random.random() < self.config["resize_prob"]:
            size = (self.config["resize_size"],) * 2
            foreground = cv2.resize(
                foreground,
                size,
                interpolation=maybe_random_interp(cv2.INTER_NEAREST, random_interp),
            )
            alpha = cv2.resize(
                alpha, size, interpolation=maybe_random_interp(cv2.INTER_NEAREST, random_interp)
            )
        sample = self.transform(
            {
                "fg": foreground,
                "alpha": alpha,
                "bg": background,
                "image_name": Path(foreground_path).name,
            }
        )
        trimap = sample["trimap"]
        trimap = torch.where(trimap < 85, 0.0, torch.where(trimap >= 170, 1.0, 0.5))
        return {"image": sample["image"], "alpha": sample["alpha"], "trimap": trimap}
