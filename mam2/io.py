"""Frame ordering and input preparation shared by all inference modes."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_frames(directory: str | Path) -> list[Path]:
    frames = [p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not frames:
        raise ValueError(f"No image frames in {directory}")
    if any(not p.stem.isdecimal() for p in frames):
        raise ValueError("Frame filenames must be numeric, e.g. 00000.jpg or 00000.png")
    frames.sort(key=lambda path: int(path.stem))
    indices = [int(path.stem) for path in frames]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate frame indices in {directory}")
    if any(right != left + 1 for left, right in zip(indices, indices[1:])):
        raise ValueError(f"Frame indices are not consecutive in {directory}")
    return frames


@contextmanager
def frame_directory(source: str):
    """Accept a numbered image folder, a single image, or a video file."""
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        yield path
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    with TemporaryDirectory(prefix="mam2_frames_") as temporary:
        directory = Path(temporary)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path))
            if image is None or not cv2.imwrite(str(directory / "00000.png"), image):
                raise OSError(f"Cannot read image: {path}")
        else:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise OSError(f"Cannot open video: {path}")
                index = 0
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    if not cv2.imwrite(str(directory / f"{index:05d}.png"), image):
                        raise OSError("Could not write temporary frame")
                    index += 1
                if index == 0:
                    raise ValueError(f"Video contains no decodable frames: {path}")
            finally:
                capture.release()
        yield directory
