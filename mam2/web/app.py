"""Local video upload, frame-by-frame prompting and asynchronous matting jobs."""

import argparse
import asyncio
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from mam2.config import load_config
from mam2.inference import MattingAnything2
from mam2.io import frame_directory, list_frames
from mam2.web.examples import VIDEO_SUFFIXES, ExampleCatalog


class Prompt(BaseModel):
    frame_index: int = Field(default=0, ge=0)
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=256)
    labels: list[int] = Field(default_factory=list, max_length=256)
    box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_prompt(self):
        if len(self.points) != len(self.labels) or any(x not in (0, 1) for x in self.labels):
            raise ValueError("Each point needs a foreground (1) or background (0) label")
        if not self.points and self.box is None:
            raise ValueError("Add a point or draw a box first")
        coordinates = [v for point in self.points for v in point] + list(self.box or ())
        if not all(np.isfinite(v) for v in coordinates):
            raise ValueError("Coordinates must be finite")
        if self.box and (self.box[0] >= self.box[2] or self.box[1] >= self.box[3]):
            raise ValueError("Draw a box with positive width and height")
        return self

    def as_model_prompt(self):
        result = {"frame_index": self.frame_index, "object_id": 1}
        if self.points:
            result.update(points=self.points, labels=self.labels)
        if self.box:
            result["box"] = self.box
        return result


class MattingOptions(BaseModel):
    resolution: str | None = None


def prepare_frames(source, directory, max_duration):
    """Decode once so seeking, prompting and final inference use identical PNGs."""
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise HTTPException(422, "Could not decode this video or its frame rate")
        if advertised_frames / fps > max_duration:
            raise HTTPException(413, f"Maximum duration: {max_duration} seconds")
        count, shape = 0, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (count + 1) / fps > max_duration:
                raise HTTPException(413, f"Maximum duration: {max_duration} seconds")
            if shape is not None and frame.shape[:2] != shape:
                raise HTTPException(422, "Frame dimensions must be constant within a video")
            shape = frame.shape[:2]
            MattingAnything2._save(directory / f"{count:05d}.png", frame)
            count += 1
        if not count:
            raise HTTPException(422, "Video contains no decodable frames")
        return fps, count, shape
    finally:
        capture.release()


def export_videos(source, alpha_directory, destination, fps, settings, progress):
    """Stream two H.264 previews; PNG alpha predictions remain the lossless output."""
    processes = []
    logs = []
    try:
        with frame_directory(str(source)) as directory:
            frames = list_frames(directory)
            alphas = list_frames(alpha_directory)
            if len(frames) != len(alphas):
                raise ValueError("The alpha sequence does not cover the complete video")
            first = cv2.imread(str(frames[0]))
            height, width = first.shape[:2]
            yy, xx = np.indices((height, width))
            cells = (
                xx // settings.checkerboard_cell_size + yy // settings.checkerboard_cell_size
            ) % 2
            colors = np.asarray(settings.checkerboard_colors, dtype=np.float32)
            checker = colors[cells][..., None]
            for name in ("alpha", "foreground"):
                log = (destination / f"{name}_encoding.log").open("wb")
                logs.append(log)
                command = [
                    settings.ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    settings.video_preset,
                    "-crf",
                    str(settings.video_crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(destination / f"{name}.mp4"),
                ]
                if settings.webm_fallback:
                    command += [
                        "-an",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v",
                        "libvpx-vp9",
                        "-row-mt",
                        "1",
                        "-deadline",
                        "realtime",
                        "-cpu-used",
                        str(settings.webm_cpu_used),
                        "-crf",
                        str(settings.webm_crf),
                        "-b:v",
                        "0",
                        "-pix_fmt",
                        "yuv420p",
                        str(destination / f"{name}.webm"),
                    ]
                processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log))
            for index, (frame, alpha_path) in enumerate(zip(frames, alphas)):
                image = cv2.imread(str(frame))
                alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
                if image is None or alpha is None or alpha.shape != (height, width):
                    raise OSError(f"Invalid output frame: {alpha_path}")
                weight = alpha.astype(np.float32)[..., None] / 255
                composite = np.round(image * weight + checker * (1 - weight)).astype(np.uint8)
                processes[0].stdin.write(cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR).tobytes())
                processes[1].stdin.write(composite.tobytes())
                progress(index + 1, len(frames))
            for process in processes:
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("Video encoding failed; see the session encoding log")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
        for log in logs:
            log.close()


def create_app(config):
    settings = config.web
    presets = dict(settings.resolution_presets)
    if (
        not presets
        or settings.default_resolution not in presets
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in presets.values())
        or settings.poll_interval_ms < 100
    ):
        raise ValueError(
            "Configure positive resolution presets, a valid default and polling >= 100 ms"
        )
    if (
        settings.checkerboard_cell_size < 1
        or len(settings.checkerboard_colors) != 2
        or any(not 0 <= color <= 255 for color in settings.checkerboard_colors)
    ):
        raise ValueError("Checkerboard needs a positive cell size and two colors in [0, 255]")
    work = Path(settings.work_directory).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    sessions = {}
    preparing_sessions = set()
    guard = threading.RLock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mam2_gpu")
    pipeline = None
    examples = ExampleCatalog(settings.get("examples"), work / "example_thumbnails")

    def model():
        nonlocal pipeline
        if pipeline is None:
            pipeline = MattingAnything2(config)
        return pipeline

    def expire():
        with guard:
            now = time.time()
            for key, session in list(sessions.items()):
                if not session["busy"] and now - session["accessed"] > settings.session_ttl_seconds:
                    shutil.rmtree(session["directory"], ignore_errors=True)
                    del sessions[key]
            # Clean up expired uploads left by a previous server process.
            for directory in work.iterdir():
                if (
                    directory.is_dir()
                    and len(directory.name) == 32
                    and all(c in "0123456789abcdef" for c in directory.name)
                    and directory.name not in sessions
                    and directory.name not in preparing_sessions
                    and now - directory.stat().st_mtime > settings.session_ttl_seconds
                ):
                    shutil.rmtree(directory, ignore_errors=True)

    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            expire()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(cleanup_loop())
        yield
        task.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Matting Anything 2", lifespan=lifespan)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    def get_session(key):
        with guard:
            if key not in sessions:
                raise HTTPException(404, "Session expired; choose a video again")
            session = sessions[key]
            session["accessed"] = time.time()
            return session

    def check_bounds(prompt, session):
        if prompt.frame_index >= session["frames"]:
            raise HTTPException(422, "Prompt frame is outside the video")
        width, height = session["width"], session["height"]
        pairs = list(prompt.points)
        if prompt.box:
            pairs += [prompt.box[:2], prompt.box[2:]]
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in pairs):
            raise HTTPException(422, "Prompt coordinates are outside the video frame")

    def check_frame(frame_index, session):
        if not 0 <= frame_index < session["frames"]:
            raise HTTPException(422, "Frame index is outside the video")

    def mask_path(session, frame_index):
        return session["directory"] / "masks" / f"{frame_index:05d}.png"

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/options")
    def options():
        return dict(
            resolution_presets=presets,
            default_resolution=settings.default_resolution,
            poll_interval_ms=settings.poll_interval_ms,
        )

    async def create_video_session(write_source, suffix=".mp4", example_id=None):
        """Prepare uploads and example selections through the same session lifecycle."""
        expire()
        key = uuid.uuid4().hex
        with guard:
            if len(sessions) + len(preparing_sessions) >= settings.max_sessions:
                raise HTTPException(429, "Demo is full; remove an old session and try again")
            preparing_sessions.add(key)
        directory = work / key
        source = directory / ("input" + (suffix if suffix in VIDEO_SUFFIXES else ".mp4"))
        try:
            directory.mkdir()
            await write_source(source)
            frames_directory = directory / "frames"
            fps, total, (height, width) = await asyncio.to_thread(
                prepare_frames, source, frames_directory, settings.max_duration_seconds
            )
            with guard:
                sessions[key] = dict(
                    directory=directory,
                    source=source,
                    frames_directory=frames_directory,
                    width=width,
                    height=height,
                    fps=fps,
                    frames=total,
                    busy=False,
                    accessed=time.time(),
                    status="ready",
                    completed=0,
                    total=total,
                    error=None,
                    prompts={},
                    revision=0,
                    example_id=example_id,
                )
                preparing_sessions.discard(key)
            return dict(
                id=key,
                width=width,
                height=height,
                fps=fps,
                frames=total,
                duration=total / fps,
                first_frame=f"/api/session/{key}/frames/0",
                example_id=example_id,
            )
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            with guard:
                preparing_sessions.discard(key)

    @app.post("/api/upload")
    async def upload(video: UploadFile = File(...)):
        async def write_source(source):
            count = 0
            with source.open("wb") as stream:
                while chunk := await video.read(1024 * 1024):
                    count += len(chunk)
                    if count > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, f"Maximum upload: {settings.max_upload_mb} MB")
                    stream.write(chunk)

        try:
            suffix = Path(video.filename or "video.mp4").suffix.lower()
            return await create_video_session(write_source, suffix)
        finally:
            await video.close()

    @app.get("/api/examples")
    def list_examples():
        return {"examples": examples.listing()}

    @app.get("/api/examples/{example_id}/thumbnail")
    def example_thumbnail(example_id: str):
        return FileResponse(examples.thumbnail(example_id), media_type="image/jpeg")

    @app.post("/api/examples/{example_id}/select")
    async def select_example(example_id: str):
        example = examples.get(example_id)
        source = example["source"]
        if source.stat().st_size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Maximum video size: {settings.max_upload_mb} MB")

        async def write_source(destination):
            await asyncio.to_thread(shutil.copyfile, source, destination)

        return await create_video_session(write_source, source.suffix.lower(), example_id)

    @app.get("/api/session/{key}/frame/{frame_index}")
    def frame_info(key: str, frame_index: int):
        session = get_session(key)
        check_frame(frame_index, session)
        with guard:
            entry = session["prompts"].get(frame_index)
            return dict(
                frame_index=frame_index,
                image=f"/api/session/{key}/frames/{frame_index}",
                prompt=entry["prompt"] if entry else None,
                foreground_pixels=entry["foreground_pixels"] if entry else 0,
                mask=(
                    f"/api/session/{key}/media/mask?frame_index={frame_index}&v={session['revision']}"
                    if entry
                    else None
                ),
            )

    @app.get("/api/session/{key}/frames/{frame_index}")
    def video_frame(key: str, frame_index: int):
        session = get_session(key)
        check_frame(frame_index, session)
        return FileResponse(session["frames_directory"] / f"{frame_index:05d}.png")

    @app.post("/api/session/{key}/prompt")
    async def preview(key: str, prompt: Prompt):
        session = get_session(key)
        check_bounds(prompt, session)
        with guard:
            if session["busy"]:
                raise HTTPException(409, "This video is already processing")
            if any(s["busy"] for s in sessions.values()):
                raise HTTPException(409, "GPU is processing a video; try again when it finishes")
            session["busy"] = True
        try:

            def predict():
                # A prompt frame is decoded independently before propagation, just as
                # in run(). Its local index is zero in this single-image preview.
                local_prompt = {**prompt.as_model_prompt(), "frame_index": 0}
                mask = model().preview(
                    session["frames_directory"] / f"{prompt.frame_index:05d}.png", local_prompt
                )
                MattingAnything2._save(mask_path(session, prompt.frame_index), mask)
                return int(np.count_nonzero(mask))

            pixels = await asyncio.get_running_loop().run_in_executor(executor, predict)
            with guard:
                session["prompts"][prompt.frame_index] = dict(
                    prompt=prompt.as_model_prompt(), foreground_pixels=pixels
                )
                session["revision"] += 1
            return dict(
                mask=f"/api/session/{key}/media/mask?frame_index={prompt.frame_index}&v={session['revision']}",
                foreground_pixels=pixels,
            )
        finally:
            with guard:
                session["busy"] = False

    @app.delete("/api/session/{key}/prompt/{frame_index}")
    def clear_prompt(key: str, frame_index: int):
        session = get_session(key)
        check_frame(frame_index, session)
        with guard:
            if session["busy"]:
                raise HTTPException(
                    409, "Wait for processing to finish before clearing this prompt"
                )
            session["prompts"].pop(frame_index, None)
            mask_path(session, frame_index).unlink(missing_ok=True)
            session["revision"] += 1
        return {"status": "cleared"}

    def run_job(session):
        def progress(stage):
            def update(done, total):
                with guard:
                    session.update(status=stage, completed=done, total=total, accessed=time.time())

            return update

        try:
            with guard:
                session.update(status="preparing", completed=0)
            model().run(
                session["frames_directory"],
                session["directory"] / "result",
                [entry["prompt"] for _, entry in sorted(session["prompts"].items())],
                progress("matting"),
                matter_max_size=presets[session["resolution"]],
                show_progress=settings.show_progress,
            )
            export_videos(
                session["frames_directory"],
                session["directory"] / "result/pha",
                session["directory"],
                session["fps"],
                settings,
                progress("encoding"),
            )
            with guard:
                session["status"] = "complete"
        except Exception as error:
            import logging

            logging.exception("Matting job failed")
            with guard:
                session.update(status="error", error=str(error))
        finally:
            with guard:
                session["busy"] = False

    @app.post("/api/session/{key}/start")
    def start(key: str, options: MattingOptions):
        session = get_session(key)
        resolution = options.resolution or settings.default_resolution
        if resolution not in presets:
            raise HTTPException(422, "Choose a configured resolution preset")
        with guard:
            if session["busy"]:
                raise HTTPException(409, "This video is already processing")
            if not any(entry["foreground_pixels"] for entry in session["prompts"].values()):
                raise HTTPException(422, "Preview a point or box selection first")
            # A new run can follow a previous result with a revised prompt.
            shutil.rmtree(session["directory"] / "result", ignore_errors=True)
            for name in ("alpha.mp4", "foreground.mp4", "alpha.webm", "foreground.webm"):
                (session["directory"] / name).unlink(missing_ok=True)
            session.update(
                busy=True, status="queued", completed=0, error=None, resolution=resolution
            )
            executor.submit(run_job, session)
        return {"status": "queued"}

    @app.get("/api/session/{key}")
    def status(key: str):
        session = get_session(key)
        with guard:
            return {
                name: session[name] for name in ("status", "completed", "total", "error", "busy")
            }

    @app.get("/api/session/{key}/media/{kind}")
    def media(key: str, kind: str, format: str = "mp4", frame_index: int = 0):
        session = get_session(key)
        check_frame(frame_index, session)
        if format not in {"mp4", "webm"}:
            raise HTTPException(422, "Video format must be mp4 or webm")
        names = {
            "first": "frames/00000.png",
            "mask": f"masks/{frame_index:05d}.png",
            "alpha": f"alpha.{format}",
            "foreground": f"foreground.{format}",
        }
        if kind not in names or not (session["directory"] / names[kind]).is_file():
            raise HTTPException(404, "Media is not ready")
        return FileResponse(
            session["directory"] / names[kind], headers={"Cache-Control": "no-store"}
        )

    @app.delete("/api/session/{key}")
    def remove(key: str):
        session = get_session(key)
        with guard:
            if session["busy"]:
                raise HTTPException(409, "Wait for processing to finish before removing this video")
            shutil.rmtree(session["directory"], ignore_errors=True)
            del sessions[key]
        return {"status": "removed"}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/web.yaml")
    parser.add_argument("overrides", nargs="*", help="YAML key=value overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    OmegaConf.resolve(config)
    if shutil.which(config.web.ffmpeg) is None:
        raise RuntimeError("Install FFmpeg with libx264 support before starting the demo")
    import uvicorn

    uvicorn.run(
        create_app(config),
        host=config.web.host,
        port=config.web.port,
        workers=1,
        access_log=config.web.access_log,
    )


if __name__ == "__main__":
    main()
