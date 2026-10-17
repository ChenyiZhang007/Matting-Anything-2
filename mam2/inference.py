"""One progressive inference pipeline for images, videos and batch manifests."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm

from mam2.config import load_config
from mam2.io import frame_directory, list_frames
from mam2.models.build import build_matter, build_predictor


class MattingAnything2:
    """PDD + MSS followed by RGB/trimap-based alpha refinement."""

    def __init__(self, config):
        self.config = config
        if config.runtime.visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(device) for device in config.runtime.visible_devices
            )
        self.device = torch.device(config.runtime.device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.runtime.precision]
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        self.predictor = build_predictor(
            config.model, config.checkpoints.main, config.lora, self.device
        )
        self.matter = build_matter(config.matter, config.checkpoints.matter, self.device).eval()

    def _add_prompt(self, state, prompt):
        frame = int(prompt.get("frame_index", 0))
        obj_id = int(prompt.get("object_id", 1))
        if not 0 <= frame < state["num_frames"]:
            raise ValueError(f"Prompt frame {frame} is outside the input video")
        mask_path = prompt.get("mask")
        if mask_path:
            if prompt.get("box") is not None or prompt.get("points") is not None:
                raise ValueError("Use a mask prompt or points/box for an object, not both")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (state["video_height"], state["video_width"]):
                raise ValueError(f"Mask must match the input frame dimensions: {mask_path}")
            return self.predictor.add_new_mask(state, frame, obj_id, mask > 0)
        points, labels = prompt.get("points"), prompt.get("labels")
        if points is not None:
            points = np.asarray(points, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int32)
            if points.ndim != 2 or points.shape[1] != 2 or labels.shape != (len(points),):
                raise ValueError("Points must be Nx2 and labels must have N values")
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("Point labels must be 0 (background) or 1 (foreground)")
        return self.predictor.add_new_points_or_box(
            state, frame, obj_id, points=points, labels=labels, box=prompt.get("box")
        )

    def preview(self, source, prompt):
        """Predict a binary mask on a prompted frame using the same main model."""
        with (
            frame_directory(str(source)) as directory,
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32),
        ):
            state = self.predictor.init_state(str(directory), offload_video_to_cpu=True)
            try:
                _, _, masks = self._add_prompt(state, prompt)
                return masks[0, 0].gt(0).to(torch.uint8).mul(255).cpu().numpy()
            finally:
                state["images"].close()
                self.predictor.reset_state(state)

    def _propagate(self, state, prompts):
        """Cover the whole video, including frames before the earliest prompt."""
        anchor = min(int(prompt.get("frame_index", 0)) for prompt in prompts)
        yield from self.predictor.propagate_in_video(
            state, start_frame_idx=anchor, show_progress=False
        )
        if anchor > 0:
            for prediction in self.predictor.propagate_in_video(
                state, start_frame_idx=anchor, reverse=True, show_progress=False
            ):
                if prediction[0] < anchor:
                    yield prediction

    def _matter_size(self, height, width, max_size=None):
        """A long-edge cap takes precedence over the legacy fixed CLI resolution."""
        if max_size is None:
            max_size = self.config.inference.get("matter_max_size")
        if max_size is not None:
            if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
                raise ValueError("MEMatte maximum size must be a positive integer")
            scale = min(1.0, max_size / max(height, width))
            return [max(1, round(height * scale)), max(1, round(width * scale))]
        resolution = self.config.inference.matter_resolution
        return list(resolution) if resolution is not None else [height, width]

    def run(
        self, source, output, prompts, progress=None, *, matter_max_size=None, show_progress=None
    ):
        """Stream lossless PNG alpha frames to disk; keep each object's trimap separate."""
        if not prompts:
            raise ValueError("At least one box, point or mask prompt is required")
        if show_progress is None:
            show_progress = self.config.runtime.get("show_progress", True)
        output = Path(output)
        if output.exists() and any(output.rglob("*.png")):
            raise FileExistsError(f"Output already contains PNG predictions: {output}")
        with frame_directory(str(source)) as directory:
            frames = list_frames(directory)
            if progress is not None:
                progress(0, len(frames))
            with (
                torch.inference_mode(),
                torch.autocast(
                    self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
                ),
            ):
                state = self.predictor.init_state(
                    str(directory),
                    offload_video_to_cpu=self.config.runtime.offload_video_to_cpu,
                    offload_state_to_cpu=self.config.runtime.offload_state_to_cpu,
                    async_loading_frames=self.config.runtime.async_loading_frames,
                )
                try:
                    for prompt in prompts:
                        self._add_prompt(state, prompt)
                    object_count = len(state["obj_ids"])
                    for completed, (index, obj_ids, masks, trimaps) in enumerate(
                        tqdm(
                            self._propagate(state, prompts),
                            total=len(frames),
                            desc=Path(source).name,
                            disable=not show_progress,
                        ),
                        start=1,
                    ):
                        bgr = cv2.imread(str(frames[index]))
                        if bgr is None:
                            raise OSError(f"Cannot read frame: {frames[index]}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                        image = image.to(self.device, dtype=torch.float32).div_(255)
                        resolution = self._matter_size(*rgb.shape[:2], matter_max_size)
                        if list(image.shape[-2:]) != resolution:
                            image = resize(image, list(resolution), antialias=True)
                        for position, obj_id in enumerate(obj_ids):
                            classes = trimaps[position : position + 1].argmax(1, keepdim=True)
                            trimap = classes.mul(128).clamp_(max=255).float().div_(255)
                            if list(trimap.shape[-2:]) != resolution:
                                trimap = resize(trimap, list(resolution), InterpolationMode.NEAREST)
                            alpha = self.matter(image, trimap)
                            if not torch.isfinite(alpha).all():
                                raise FloatingPointError(f"Non-finite alpha at frame {index}")
                            alpha = (
                                alpha.mul(255).clamp_(0, 255).to(torch.uint8)[0, 0].cpu().numpy()
                            )
                            if alpha.shape != rgb.shape[:2]:
                                alpha = cv2.resize(
                                    alpha,
                                    (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            folder = output if object_count == 1 else output / f"object_{obj_id}"
                            name = frames[index].stem + ".png"
                            self._save(folder / "pha" / name, alpha)
                            if self.config.output.save_trimaps:
                                tri = (
                                    trimaps[position]
                                    .argmax(0)
                                    .mul(128)
                                    .clamp_(max=255)
                                    .to(torch.uint8)
                                    .cpu()
                                    .numpy()
                                )
                                self._save(folder / "trimap" / name, tri)
                            if self.config.output.save_masks:
                                mask = (
                                    masks[position, 0].gt(0).to(torch.uint8).mul(255).cpu().numpy()
                                )
                                self._save(folder / "mask" / name, mask)
                        if progress is not None:
                            progress(completed, len(frames))
                finally:
                    state["images"].close()
                    self.predictor.reset_state(state)

    @staticmethod
    def _save(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), array):
            raise OSError(f"Could not save prediction: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference YAML")
    parser.add_argument("overrides", nargs="*", help="YAML key=value overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    OmegaConf.resolve(config)
    if config.input.manifest:
        jobs = OmegaConf.to_container(
            OmegaConf.load(config.input.manifest), resolve=True, throw_on_missing=True
        )
        if not isinstance(jobs, list):
            raise ValueError("The batch manifest must be a YAML list of input jobs")
    else:
        jobs = [
            {
                "source": config.input.source,
                "name": config.input.name,
                "prompts": OmegaConf.to_container(
                    config.input.prompts, resolve=True, throw_on_missing=True
                ),
            }
        ]
    pipeline = MattingAnything2(config)
    for job in jobs:
        name = job["name"]
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Job names must be nonempty relative output paths")
        pipeline.run(job["source"], Path(config.output.directory) / name, job["prompts"])


if __name__ == "__main__":
    main()
