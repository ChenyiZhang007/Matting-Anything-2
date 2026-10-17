# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from mam2.training.dataset.vos_segment_loader import (
    MattePNGSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False
    alpha_path: Optional[str] = None
    background_key: Optional[str] = None


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class VideoMattingRawDataset(VOSRawDataset):
    """Read per-video foreground/alpha folders without rewriting path strings."""

    def __init__(
        self,
        fg_folder,
        bg_folder,
        file_list_txt=None,
        sample_rate=1,
        foreground_subdirectory="img",
        alpha_subdirectory="alpha",
        image_extensions=(".jpg", ".jpeg", ".png"),
        alpha_extension=".png",
    ):
        self.fg_folder = Path(fg_folder)
        self.bg_folder = bg_folder
        self.sample_rate = sample_rate
        self.foreground_subdirectory = foreground_subdirectory
        self.alpha_subdirectory = alpha_subdirectory
        self.image_extensions = set(image_extensions)
        self.alpha_extension = alpha_extension
        if file_list_txt:
            self.video_names = [
                line.strip()
                for line in Path(file_list_txt).read_text().splitlines()
                if line.strip()
            ]
        else:
            self.video_names = sorted(p.name for p in self.fg_folder.iterdir() if p.is_dir())
        if not self.video_names or sample_rate < 1:
            raise ValueError("No foreground videos or invalid sample_rate")

    def __len__(self):
        return len(self.video_names)

    def get_video(self, index):
        name = self.video_names[index]
        root = self.fg_folder / name
        alpha_root = root / self.alpha_subdirectory
        paths = sorted(
            (
                p
                for p in (root / self.foreground_subdirectory).iterdir()
                if p.suffix.lower() in self.image_extensions
            ),
            key=lambda p: int(p.stem),
        )
        frames = [
            VOSFrame(
                int(p.stem),
                str(p),
                alpha_path=str(alpha_root / (p.stem + self.alpha_extension)),
                background_key=str(Path(name) / (p.stem + self.alpha_extension)),
            )
            for p in paths[:: self.sample_rate]
        ]
        return VOSVideo(name, index, frames), MattePNGSegmentLoader(str(alpha_root))


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(self.img_folder, os.path.dirname(video_name))
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(video_mask_root, self.single_object_mode)

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
