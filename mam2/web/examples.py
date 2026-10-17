"""Configured, read-only example videos and their cached preview thumbnails."""

import hashlib
import re
import sys
import threading
import uuid
from pathlib import Path

import cv2
from fastapi import HTTPException

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class ExampleCatalog:
    """Resolve examples from a source checkout or installed wheel data directory."""

    def __init__(self, settings, cache_directory):
        self.cache_directory = Path(cache_directory)
        self.videos = {}
        self.locks = {}
        config = settings or {}
        self.thumbnail_size = config.get("thumbnail_size", 320)
        if (
            isinstance(self.thumbnail_size, bool)
            or not isinstance(self.thumbnail_size, int)
            or self.thumbnail_size < 1
        ):
            raise ValueError("Example thumbnail size must be a positive integer")
        videos = config.get("videos", [])
        if not videos:
            return
        directory = Path(config.get("directory", "assets")).expanduser()
        if not directory.is_absolute():
            package_root = Path(__file__).resolve().parents[2]
            candidates = [
                package_root / directory,
                package_root / "share/mam2" / directory,
                Path(sys.prefix) / "share/mam2" / directory,
            ]
            directory = next((path for path in candidates if path.is_dir()), candidates[0])
        directory = directory.resolve()
        for video in videos:
            identifier = video["id"]
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
                raise ValueError(
                    "Example IDs may contain letters, numbers, underscores and hyphens"
                )
            if identifier in self.videos:
                raise ValueError(f"Duplicate example ID: {identifier}")
            source = (directory / video["video"]).resolve()
            if not source.is_relative_to(directory):
                raise ValueError("Example videos must be inside the configured example directory")
            if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError(f"Example video is missing or unsupported: {source}")
            self.videos[identifier] = dict(
                id=identifier, name=str(video.get("name", identifier)), source=source
            )
            self.locks[identifier] = threading.Lock()

    def get(self, identifier):
        video = self.videos.get(identifier)
        if video is None or not video["source"].is_file():
            raise HTTPException(404, "Example video is unavailable")
        return video

    def fingerprint(self, video):
        source = video["source"]
        stat = source.stat()
        value = f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{self.thumbnail_size}"
        return hashlib.sha256(value.encode()).hexdigest()

    def listing(self):
        return [
            dict(
                id=video["id"],
                name=video["name"],
                thumbnail=(f"/api/examples/{video['id']}/thumbnail?v={self.fingerprint(video)}"),
            )
            for video in self.videos.values()
            if video["source"].is_file()
        ]

    def thumbnail(self, identifier):
        video = self.get(identifier)
        destination = self.cache_directory / f"{self.fingerprint(video)}.jpg"
        with self.locks[identifier]:
            if destination.is_file():
                return destination
            capture = cv2.VideoCapture(str(video["source"]))
            try:
                ok, image = capture.read()
            finally:
                capture.release()
            if not ok:
                raise HTTPException(422, "Could not decode this example video")
            height, width = image.shape[:2]
            scale = min(1.0, self.thumbnail_size / max(height, width))
            if scale < 1:
                image = cv2.resize(
                    image,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.jpg")
            try:
                if not cv2.imwrite(str(temporary), image):
                    raise OSError("Could not save the example thumbnail")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
