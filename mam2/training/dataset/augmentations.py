"""Training augmentations adapted from MEMatte / GCA-Matting.

Only the image-matting training transforms are retained. User settings are passed
explicitly from YAML; there is no mutable module-global configuration.
"""

import logging
import math
import numbers
import random

import cv2
import numpy as np
import torch

INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(interpolation, enabled):
    return np.random.choice(INTERPOLATIONS) if enabled else interpolation


class ToTensor(object):
    """
    Convert BGR arrays to RGB tensors in [0, 1]; the model normalizes RGB.
    """

    def __init__(self, phase="test"):
        self.phase = phase

    def __call__(self, sample):
        (image, alpha, trimap, mask) = (
            sample["image"][:, :, ::-1],
            sample["alpha"],
            sample["trimap"],
            sample["mask"],
        )
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        mask = np.expand_dims(mask.astype(np.float32), axis=0)
        image /= 255.0
        if self.phase == "train":
            fg = sample["fg"][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
            sample["fg"] = torch.from_numpy(fg)
            bg = sample["bg"][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
            sample["bg"] = torch.from_numpy(bg)
        (sample["image"], sample["alpha"], sample["trimap"]) = (
            torch.from_numpy(image),
            torch.from_numpy(alpha),
            torch.from_numpy(trimap).to(torch.long),
        )
        sample["trimap"] = sample["trimap"][None, ...].float()
        sample["mask"] = torch.from_numpy(mask).float()
        return sample


class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        flip=None,
        resample=False,
        fillcolor=0,
        *,
        random_interpolation=True,
    ):
        self.random_interpolation = random_interpolation
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, (
                "degrees should be a list or tuple and it must be of length 2."
            )
            self.degrees = degrees
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, (
                "translate should be a list or tuple and it must be of length 2."
            )
            for t in translate:
                if not 0.0 <= t <= 1.0:
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, (
                "scale should be a list or tuple and it must be of length 2."
            )
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, (
                    "shear should be a list or tuple and it must be of length 2."
                )
                self.shear = shear
        else:
            self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (
                np.round(random.uniform(-max_dx, max_dx)),
                np.round(random.uniform(-max_dy, max_dy)),
            )
        else:
            translations = (0, 0)
        if scale_ranges is not None:
            scale = (
                random.uniform(scale_ranges[0], scale_ranges[1]),
                random.uniform(scale_ranges[0], scale_ranges[1]),
            )
        else:
            scale = (1.0, 1.0)
        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0
        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(int) * 2 - 1
        return (angle, translations, scale, shear, flip)

    def __call__(self, sample):
        (fg, alpha) = (sample["fg"], sample["alpha"])
        (rows, cols, ch) = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params(
                (0, 0), self.translate, self.scale, self.shear, self.flip, fg.size
            )
        else:
            params = self.get_params(
                self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size
            )
        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))
        fg = cv2.warpAffine(
            fg,
            M,
            (cols, rows),
            flags=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation)
            + cv2.WARP_INVERSE_MAP,
        )
        alpha = cv2.warpAffine(
            alpha,
            M,
            (cols, rows),
            flags=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation)
            + cv2.WARP_INVERSE_MAP,
        )
        (sample["fg"], sample["alpha"]) = (fg, alpha)
        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x,
            math.sin(angle + shear) * scale_x,
            0,
            -math.sin(angle) * scale_y,
            math.cos(angle + shear) * scale_y,
            0,
        ]
        matrix = [m / d for m in matrix]
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (
            -center[1] - translate[1]
        )
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (
            -center[1] - translate[1]
        )
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        (fg, alpha) = (sample["fg"], sample["alpha"])
        if np.all(alpha == 0):
            return sample_ori
        fg = cv2.cvtColor(fg.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand() * (1.1 - val_bar) / 5 - (1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample["fg"] = fg * 255
        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, *, random_interpolation=True):
        self.random_interpolation = random_interpolation
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        (fg, alpha, trimap, mask, name) = (
            sample["fg"],
            sample["alpha"],
            sample["trimap"],
            sample["mask"],
            sample["image_name"],
        )
        bg = sample["bg"]
        (h, w) = trimap.shape
        bg = cv2.resize(
            bg,
            (w, h),
            interpolation=maybe_random_interp(cv2.INTER_CUBIC, self.random_interpolation),
        )
        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                fg = cv2.resize(
                    fg,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation),
                )
                alpha = cv2.resize(
                    alpha,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation),
                )
                trimap = cv2.resize(
                    trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST
                )
                bg = cv2.resize(
                    bg,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=maybe_random_interp(cv2.INTER_CUBIC, self.random_interpolation),
                )
                mask = cv2.resize(
                    mask, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST
                )
                (h, w) = trimap.shape
        small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(
            zip(
                *np.where(
                    small_trimap[
                        self.margin // 4 : (h - self.margin) // 4,
                        self.margin // 4 : (w - self.margin) // 4,
                    ]
                    == 128
                )
            )
        )
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (
                np.random.randint(0, h - self.output_size[0] + 1),
                np.random.randint(0, w - self.output_size[1] + 1),
            )
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)
        fg_crop = fg[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
            :,
        ]
        alpha_crop = alpha[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
        ]
        bg_crop = bg[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
            :,
        ]
        trimap_crop = trimap[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
        ]
        mask_crop = mask[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
        ]
        if len(np.where(trimap == 128)[0]) == 0:
            self.logger.error(
                "{} does not have enough unknown area for crop. Resized to target size.left_top: {}".format(
                    name, left_top
                )
            )
            fg_crop = cv2.resize(
                fg,
                self.output_size[::-1],
                interpolation=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation),
            )
            alpha_crop = cv2.resize(
                alpha,
                self.output_size[::-1],
                interpolation=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation),
            )
            trimap_crop = cv2.resize(
                trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST
            )
            bg_crop = cv2.resize(
                bg,
                self.output_size[::-1],
                interpolation=maybe_random_interp(cv2.INTER_CUBIC, self.random_interpolation),
            )
            mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        sample.update(
            {
                "fg": fg_crop,
                "alpha": alpha_crop,
                "trimap": trimap_crop,
                "mask": mask_crop,
                "bg": bg_crop,
            }
        )
        return sample


class GenMask(object):
    def __init__(self, *, random_interpolation=True):
        self.random_interpolation = random_interpolation
        self.erosion_kernels = [None] + [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)
        ]

    def __call__(self, sample):
        alpha_ori = sample["alpha"]
        (h, w) = alpha_ori.shape
        max_kernel_size = 30
        alpha = cv2.resize(
            alpha_ori,
            (640, 640),
            interpolation=maybe_random_interp(cv2.INTER_NEAREST, self.random_interpolation),
        )
        fg_mask = (alpha + 1e-05).astype(int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-05).astype(int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-05).astype(int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-05).astype(int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])
        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0
        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample["trimap"] = trimap
        low = 0.01
        high = 1.0
        thres = random.random() * (high - low) + low
        seg_mask = (alpha >= thres).astype(int).astype(np.uint8)
        random_num = random.randint(0, 3)
        if random_num == 0:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
        elif random_num == 1:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
        elif random_num == 2:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
        elif random_num == 3:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)]
            )
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample["mask"] = seg_mask
        return sample


class Composite(object):
    def __call__(self, sample):
        (fg, bg, alpha) = (sample["fg"], sample["bg"], sample["alpha"])
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255
        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample["image"] = image
        return sample


class CutMask(object):
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample
        mask = sample["mask"]
        (h, w) = mask.shape
        (perturb_size_h, perturb_size_w) = (
            random.randint(h // 4, h // 2),
            random.randint(w // 4, w // 2),
        )
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)
        mask[x : x + perturb_size_h, y : y + perturb_size_w] = mask[
            x1 : x1 + perturb_size_h, y1 : y1 + perturb_size_w
        ].copy()
        sample["mask"] = mask
        return sample
