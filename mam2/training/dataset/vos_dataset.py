# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared frame construction with explicit mask/trimap supervision by data type."""

import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from mam2.training.utils.data_utils import Frame, Object, VideoDatapoint


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def load_alpha(path):
    with Image.open(path) as image:
        return np.array(image.convert("L"))


class VOSDataset(VisionDataset):
    """Apply temporal transforms after loading the sampled annotated frames."""

    def __init__(
        self,
        transforms,
        training,
        video_dataset,
        sampler,
        multiplier,
        always_target=True,
        target_segments_available=True,
        max_retries=100,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler
        self.repeat_factors = torch.full((len(video_dataset),), float(multiplier))
        self.curr_epoch = 0
        self.always_target = always_target
        self.target_segments_available = target_segments_available
        self.max_retries = max_retries
        logging.info("%s: %d samples", type(self).__name__, len(video_dataset))

    def __len__(self):
        return len(self.video_dataset)

    def __getitem__(self, index):
        attempts = self.max_retries if self.training else 1
        for attempt in range(attempts):
            try:
                datapoint = self._load(int(index))
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)
                return datapoint
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                if attempt + 1 == attempts:
                    raise RuntimeError(
                        f"Cannot load {type(self).__name__} sample {index}"
                    ) from error
                logging.warning("Sample %s failed: %s; retrying", index, error)
                index = random.randrange(len(self))

    def _frame_image(self, frame):
        if frame.data is None:
            return load_rgb(frame.image_path)
        pixels = frame.data.cpu().numpy().transpose(1, 2, 0) * 255
        return Image.fromarray(pixels.astype(np.uint8))

    def _load(self, index):
        video, segment_loader = self.video_dataset.get_video(index)
        sample = self.sampler.sample(video, segment_loader, epoch=self.curr_epoch)
        frames = []
        for raw_frame in sample.frames:
            image = self._frame_image(raw_frame)
            width, height = image.size
            segments = segment_loader.load(raw_frame.frame_idx)
            objects = []
            for object_id in sample.object_ids:
                if object_id in segments:
                    segment = segments[object_id].to(torch.uint8)
                elif self.always_target:
                    segment = torch.zeros(height, width, dtype=torch.uint8)
                else:
                    continue
                objects.append(Object(object_id, raw_frame.frame_idx, segment, None))
            frames.append(Frame(image, objects))
        return VideoDatapoint(frames, video.video_id, (height, width))


class VideoMattingDataset(VOSDataset):
    """Composite a foreground video using an explicit frame/background mapping."""

    def __init__(self, *args, bg_dict_path, **kwargs):
        super().__init__(*args, **kwargs)
        with open(bg_dict_path) as stream:
            self.backgrounds = json.load(stream)

    def _frame_image(self, frame):
        foreground = np.array(load_rgb(frame.image_path))
        alpha = load_alpha(frame.alpha_path).astype(np.float64)[..., None] / 255
        background_path = (
            Path(self.video_dataset.bg_folder) / self.backgrounds[frame.background_key]
        )
        background = np.array(load_rgb(background_path))
        if background.shape != foreground.shape:
            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        composite = foreground * alpha + background * (1 - alpha)
        return Image.fromarray(composite.astype(np.uint8))


class SegmentationDataset(VOSDataset):
    """Single-image binary masks; use transforms without GenTrimap."""

    def _load(self, index):
        image_path, alpha_path = self.video_dataset.get_sample(index)
        return self._datapoint(load_rgb(image_path), load_alpha(alpha_path), index)

    @staticmethod
    def _datapoint(image, alpha, index):
        width, height = image.size
        if alpha.shape != (height, width):
            raise ValueError("Image and annotation dimensions differ")
        frame = Frame(image, [Object(0, 0, torch.from_numpy(alpha).to(torch.uint8), None)])
        return VideoDatapoint([frame], index, (height, width))


class MattingDataset(SegmentationDataset):
    """Image matting with per-source composition settings supplied through YAML."""

    def _load(self, index):
        image_path, alpha_path, background_path, options = self.video_dataset.get_sample(index)
        image = load_rgb(image_path)
        matte = load_alpha(alpha_path)
        if np.random.random() >= options["probability"]:
            return self._datapoint(image, matte, index)
        foreground = np.array(image)
        height, width = matte.shape
        if foreground.shape[:2] != matte.shape:
            raise ValueError("Foreground and alpha dimensions differ")
        background = cv2.resize(np.array(load_rgb(background_path)), (width, height))
        scale = np.random.uniform(*options["scale_range"])
        if not 0 < scale <= 1:
            raise ValueError("Foreground composition scale must be in (0, 1]")
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        foreground = cv2.resize(foreground, size, interpolation=cv2.INTER_LINEAR)
        matte = cv2.resize(matte, size, interpolation=cv2.INTER_LINEAR)
        x = np.random.randint(0, width - size[0] + 1)
        y = np.random.randint(0, height - size[1] + 1)
        region = np.s_[y : y + size[1], x : x + size[0]]
        alpha = matte[..., None] / 255
        background[region] = foreground * alpha + background[region] * (1 - alpha)
        target = np.zeros((height, width), dtype=np.uint8)
        target[region] = matte
        return self._datapoint(Image.fromarray(background), target, index)
