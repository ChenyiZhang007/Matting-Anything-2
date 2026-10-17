"""Batch-convert VM108 RGBA frames using the bundled foreground/background correspondence."""

import argparse
import filecmp
import hashlib
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from threading import Event

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from mam2.config import load_config


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Expected a relative POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Expected a normalized relative path: {value!r}")
    return path


def _split(path):
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Split must contain unique video names: {path}")
    for name in names:
        _relative(name)
    return names


def _rgba_directory(root, subdirectory):
    if subdirectory != "auto":
        return root if subdirectory == "." else root / str(_relative(subdirectory))
    candidates = []
    for directory in [root, root / "rgba"]:
        if directory.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".png" for p in directory.iterdir()
        ):
            candidates.append(directory)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected PNG frames in {root} or its rgba/ subdirectory; set rgba_subdirectory explicitly"
        )
    return candidates[0]


def plan_conversion(config):
    paths, options = config.paths, config.preparation
    for name in ["workers", "png_compression"]:
        if type(options[name]) is not int:
            raise ValueError(f"preparation.{name} must be an integer")
    if options.workers < 1 or not 0 <= options.png_compression <= 9:
        raise ValueError("Use workers >= 1 and png_compression in [0, 9]")
    if type(options.resume) is not bool or type(options.dry_run) is not bool:
        raise ValueError("resume and dry_run must be booleans")
    if options.split not in {"all", "train", "val"}:
        raise ValueError("preparation.split must be all, train or val")
    if options.background_mode not in {"copy", "symlink"}:
        raise ValueError("preparation.background_mode must be copy or symlink")
    source = Path(paths.source_root).expanduser().resolve()
    output = Path(paths.output_root).expanduser().resolve()
    foregrounds = Path(paths.foregrounds).expanduser().resolve()
    backgrounds = Path(paths.backgrounds).expanduser().resolve()
    destination_fg = output / "FG/FG_done"
    resolved_destination = destination_fg.resolve()
    if (
        source == output
        or destination_fg.is_symlink()
        or foregrounds == resolved_destination
        or foregrounds in resolved_destination.parents
        or resolved_destination in foregrounds.parents
    ):
        raise ValueError("Output foregrounds must be separate from the original dataset")
    metadata = {
        "frame_corr.json": Path(paths.correspondence).expanduser().resolve(),
        "train_videos.txt": Path(paths.train_split).expanduser().resolve(),
        "val_videos.txt": Path(paths.val_split).expanduser().resolve(),
    }
    mapping = json.loads(metadata["frame_corr.json"].read_text())
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Correspondence must be a nonempty foreground-to-background mapping")
    train, val = _split(metadata["train_videos.txt"]), _split(metadata["val_videos.txt"])
    if set(train) & set(val):
        raise ValueError("Training and validation video lists overlap")
    grouped = defaultdict(list)
    for key, value in mapping.items():
        frame, background = _relative(key), _relative(value)
        if frame.suffix != ".png" or not frame.stem.isdecimal() or len(frame.parts) < 2:
            raise ValueError(f"Invalid foreground frame key: {key}")
        grouped[str(frame.parent)].append((frame.name, str(background)))
    if set(grouped) != set(train) | set(val):
        raise ValueError("Correspondence videos must match the train/validation lists")
    selected = (
        train + val if options.split == "all" else (train if options.split == "train" else val)
    )
    jobs, background_names = [], set()
    for video in selected:
        frames = sorted(grouped[video], key=lambda pair: int(Path(pair[0]).stem))
        indices = [int(Path(name).stem) for name, _ in frames]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate numeric frame IDs: {video}")
        directory = _rgba_directory(foregrounds / video, options.rgba_subdirectory)
        expected = {name for name, _ in frames}
        for name, background in frames:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
            background_names.add(background)
        target = destination_fg / video
        for folder in [target / "img", target / "alpha"]:
            existing = (
                {p.name for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
                if folder.is_dir()
                else set()
            )
            if existing - expected:
                raise ValueError(f"Output contains unmapped frames: {folder}")
            if existing and not options.resume:
                raise FileExistsError(
                    f"Output exists: {folder}; use preparation.resume=true to verify and resume"
                )
        jobs.append(
            dict(video=video, source=directory, target=target, frames=[name for name, _ in frames])
        )
    for name in sorted(background_names):
        if not (backgrounds / name).is_file():
            raise FileNotFoundError(backgrounds / name)
    destination_bg = output / "BG/BG_done2"
    if destination_bg.is_symlink():
        if options.background_mode != "symlink" or destination_bg.resolve() != backgrounds:
            raise FileExistsError(f"Background link points to a different source: {destination_bg}")
    elif destination_bg.exists():
        if options.background_mode == "symlink" or destination_bg.resolve() == backgrounds:
            raise FileExistsError(
                f"Background destination conflicts with the source: {destination_bg}"
            )
        for name in background_names:
            target = destination_bg / name
            if target.exists() and not filecmp.cmp(backgrounds / name, target, shallow=False):
                raise FileExistsError(f"Existing background differs from source: {target}")
    for name, path in metadata.items():
        target = output / name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise FileExistsError(
                f"Existing metadata differs from the selected mapping/splits: {target}"
            )
    return dict(
        output=output,
        jobs=jobs,
        backgrounds=backgrounds,
        background_names=sorted(background_names),
        metadata=metadata,
    )


def _write_png(path, pixels, compression):
    if path.exists():
        previous = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if previous is None or not np.array_equal(previous, pixels):
            raise FileExistsError(f"Existing output pixels differ from source: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not success:
        raise OSError(f"Cannot encode PNG: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _convert_video(job, compression, progress, stop):
    written = verified = 0
    for name in job["frames"]:
        if stop.is_set():
            return None
        path = job["source"] / name
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Expected an 8-bit RGBA PNG: {path}")
        # OpenCV reads BGRA; writing its BGR channels preserves the original RGB values.
        rgb_written = _write_png(job["target"] / "img" / name, rgba[..., :3], compression)
        alpha_written = _write_png(job["target"] / "alpha" / name, rgba[..., 3], compression)
        written += rgb_written or alpha_written
        verified += not (rgb_written or alpha_written)
        progress(1)
    return dict(video=job["video"], frames=len(job["frames"]), written=written, verified=verified)


def _copy_file(source, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(config):
    plan = plan_conversion(config)
    options = config.preparation
    summary = dict(
        videos=len(plan["jobs"]),
        frames=sum(len(job["frames"]) for job in plan["jobs"]),
        background_frames=len(plan["background_names"]),
        output=str(plan["output"]),
        split=options.split,
        background_mode=options.background_mode,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if options.dry_run:
        print("Dry run passed; no files were written.")
        return summary
    cv2.setNumThreads(1)
    stop, results = Event(), []
    with tqdm(total=summary["frames"], desc="VM108", unit="frame") as progress:
        with ThreadPoolExecutor(max_workers=options.workers) as executor:
            futures = [
                executor.submit(_convert_video, job, options.png_compression, progress.update, stop)
                for job in plan["jobs"]
            ]
            try:
                for future in as_completed(futures):
                    results.append(future.result())
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
    destination_bg = plan["output"] / "BG/BG_done2"
    destination_bg.parent.mkdir(parents=True, exist_ok=True)
    if options.background_mode == "symlink":
        if not destination_bg.is_symlink():
            destination_bg.symlink_to(plan["backgrounds"], target_is_directory=True)
    else:
        for name in tqdm(plan["background_names"], desc="Backgrounds", unit="frame"):
            _copy_file(plan["backgrounds"] / name, destination_bg / name)
    for name, source in plan["metadata"].items():
        _copy_file(source, plan["output"] / name)
    report = dict(
        **summary,
        written_frames=sum(row["written"] for row in results),
        verified_frames=sum(row["verified"] for row in results),
        metadata_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in plan["metadata"].items()
        },
        per_video=sorted(results, key=lambda row: row["video"]),
    )
    (plan["output"] / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {summary['frames']} frames in {plan['output']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="VM108 preparation YAML")
    args, overrides = parser.parse_known_args()
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("Configuration overrides must use key=value")
    config = load_config(args.config, overrides)
    OmegaConf.resolve(config)
    prepare(config)


if __name__ == "__main__":
    main()
