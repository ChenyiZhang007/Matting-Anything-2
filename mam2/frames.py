"""Bounded frame loading for progressive video inference."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from mam2.io import list_frames


class VideoFrames:
    """Cache two normalized frames and optionally prefetch the next one."""

    def __init__(self, directory, image_size, mean, std, device, prefetch=False):
        self.paths = list_frames(directory)
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.device = device
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending = None
        with Image.open(self.paths[0]) as image:
            self.video_width, self.video_height = image.size

    def _read(self, index):
        with Image.open(self.paths[index]) as image:
            if image.size != (self.video_width, self.video_height):
                raise ValueError(f"Frame size changed within the video: {self.paths[index]}")
            # Keep the research code's normalization order for checkpoint parity.
            array = np.array(image.convert("RGB").resize((self.image_size,) * 2)) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.pending is not None and self.pending[0] == index:
            tensor = self.pending[1].result()
            self.pending = None
        else:
            tensor = self._read(index)
        self.cache[index] = tensor
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        next_index = index + 1
        if self.executor is not None and next_index < len(self) and next_index not in self.cache:
            if self.pending is not None:
                self.pending[1].cancel()
            self.pending = (next_index, self.executor.submit(self._read, next_index))
        return tensor

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.cache.clear()
