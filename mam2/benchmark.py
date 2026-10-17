"""Run YouTubeMatte inference or score lossless alpha PNGs with one YAML configuration."""

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from mam2.config import load_config
from mam2.io import list_frames
from mam2.metrics import AlphaMetrics


def _paired_frames(left, right):
    first, second = list_frames(left), list_frames(right)
    if [p.stem for p in first] != [p.stem for p in second]:
        raise ValueError(f"Frame stems differ: {left} and {right}")
    return first, second


def _validate_images(paths, resolution, *, prediction=False):
    for path in paths:
        with Image.open(path) as image:
            if image.size != tuple(reversed(resolution)):
                raise ValueError(
                    f"Expected native H×W {resolution}, got {image.size[::-1]}: {path}"
                )
            if prediction and (image.format != "PNG" or image.mode != "L"):
                raise ValueError(f"Predictions must be 8-bit grayscale PNGs: {path}")


def _mask_prompt(path, resolution, *, validate):
    if validate:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot read first-frame mask: {path}")
        # The official archive stores binary masks as three identical RGB channels.
        if mask.ndim == 3 and mask.shape[2] == 3 and np.all(mask == mask[..., :1]):
            mask = mask[..., 0]
        if mask.shape != tuple(resolution):
            raise ValueError(f"First-frame mask must match native H×W {resolution}: {path}")
        if mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"First-frame mask must be 8-bit binary (0/1 or 0/255): {path}")
        if not mask.any():
            raise ValueError(f"First-frame mask has no foreground: {path}")
    return [dict(object_id=1, frame_index=0, mask=str(path))]


def benchmark_jobs(config, *, validate_masks=True):
    """Use the configured inventory so missing clips cannot silently change the average."""
    settings = config.benchmark
    root = Path(settings.data_root).expanduser().resolve()
    prompts = OmegaConf.to_container(settings.prompts, resolve=True)
    mode = settings.get("prompt_mode", "box_click")
    if mode not in {"box_click", "mask"}:
        raise ValueError("benchmark.prompt_mode must be 'box_click' or 'mask'")
    mask_root = None
    if mode == "mask":
        if not settings.get("first_frame_mask_root"):
            raise ValueError("Set benchmark.first_frame_mask_root for mask-guided inference")
        mask_root = Path(settings.first_frame_mask_root).expanduser().resolve()
    selected = list(prompts) if settings.clips is None else list(settings.clips)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one clip, without duplicate names")
    jobs = []
    for name in selected:
        if (
            name not in prompts
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or len(Path(name).parts) != 2
        ):
            raise ValueError(f"Unknown YouTubeMatte clip: {name}")
        source = root / name / settings.rgb_directory
        truth = root / name / settings.alpha_directory
        rgb_frames, truth_frames = _paired_frames(source, truth)
        if len(rgb_frames) != settings.frames_per_clip:
            raise ValueError(f"Expected {settings.frames_per_clip} frames in {source}")
        _validate_images(rgb_frames + truth_frames, settings.resolution)
        guidance = (
            _mask_prompt(
                mask_root / (Path(name).name + ".png"), settings.resolution, validate=validate_masks
            )
            if mode == "mask"
            else prompts[name]
        )
        jobs.append(dict(name=name, source=source, truth=truth_frames, prompts=guidance))
    return jobs


def infer(config, jobs):
    # Import and build the models only for inference; scoring needs no checkpoints.
    from mam2.inference import MattingAnything2

    root = Path(config.output.directory).expanduser().resolve()
    # Preflight every output before beginning a long run.
    for job in jobs:
        folder = root / job["name"]
        if folder.exists() and any(folder.rglob("*.png")):
            raise FileExistsError(f"Output already contains predictions: {folder}")
    pipeline = MattingAnything2(config)
    for job in jobs:
        # Reset per clip so subset/full runs use the same random initialization.
        seed = int(config.runtime.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pipeline.run(job["source"], root / job["name"], job["prompts"])
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, root / "inference_config.yaml", resolve=True)


def _read_alpha(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot read alpha: {path}")
    return torch.from_numpy(image).to(device=device, dtype=torch.float32).div_(255)


def _mean(records):
    return {
        name: sum(record["metrics"][name] for record in records) / len(records)
        for name in AlphaMetrics.names
    }


@torch.inference_mode()
def evaluate(config, jobs):
    root = Path(config.output.directory).expanduser().resolve()
    device = torch.device(config.benchmark.metrics_device)
    # Validate all predictions before scoring; do not skip missing or invalid frames.
    for job in jobs:
        predicted, truth = _paired_frames(root / job["name"] / "pha", job["truth"][0].parent)
        _validate_images(predicted, config.benchmark.resolution, prediction=True)
        job["predicted"], job["truth"] = predicted, truth
    metric = AlphaMetrics(device, float(config.benchmark.gradient_sigma))
    records = []
    # Explicitly disable autocast even when called from a mixed-precision context.
    with torch.autocast(device.type, enabled=False):
        for job in jobs:
            values = {name: [] for name in AlphaMetrics.names}
            previous_pred = previous_truth = None
            for pred_path, truth_path in tqdm(
                zip(job["predicted"], job["truth"]),
                total=len(job["truth"]),
                desc=job["name"],
                disable=not config.runtime.show_progress,
            ):
                pred, truth = _read_alpha(pred_path, device), _read_alpha(truth_path, device)
                current = metric(pred, truth, previous_pred, previous_truth)
                for name, value in current.items():
                    number = value.item()
                    if not math.isfinite(number):
                        raise FloatingPointError(f"Non-finite {name}: {pred_path}")
                    values[name].append(number)
                previous_pred, previous_truth = pred, truth
            records.append(
                dict(
                    clip=job["name"],
                    frames=len(job["truth"]),
                    frame_stems=[path.stem for path in job["truth"]],
                    metrics={name: sum(items) / len(items) for name, items in values.items()},
                    per_frame=values,
                )
            )
    summary = dict(
        benchmark="YouTubeMatte",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        videos=len(records),
        frames=sum(record["frames"] for record in records),
        resolution=list(config.benchmark.resolution),
        aggregation="Mean frames within each clip, then mean clips equally",
        first_frame_dtssd="Zero, included in each clip mean",
        precision="float32",
        gradient_sigma=float(config.benchmark.gradient_sigma),
        overall=_mean(records),
        by_dataset={
            group: _mean([r for r in records if r["clip"].split("/")[0] == group])
            for group in sorted({r["clip"].split("/")[0] for r in records})
        },
        clips=records,
    )
    (root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "metrics_per_clip.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip", "frames", *AlphaMetrics.names])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {"clip": record["clip"], "frames": record["frames"], **record["metrics"]}
            )
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved {root / 'metrics.json'} and {root / 'metrics_per_clip.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["infer", "evaluate"])
    parser.add_argument("--config", required=True, help="YouTubeMatte benchmark YAML")
    args, overrides = parser.parse_known_args()
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("Configuration overrides must use key=value")
    config = load_config(args.config, overrides)
    OmegaConf.resolve(config)
    if config.runtime.visible_devices is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.runtime.visible_devices))
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    jobs = benchmark_jobs(config, validate_masks=args.action == "infer")
    if args.action == "infer":
        infer(config, jobs)
    else:
        evaluate(config, jobs)


if __name__ == "__main__":
    main()
