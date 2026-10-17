"""Download and verify checkpoints for inference and model initialization."""

import argparse
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from omegaconf import OmegaConf

from mam2.config import load_config


def resolve_checkpoint(source: str | Path | Mapping) -> Path:
    """Download a configured Hub file once, or use an existing local checkpoint."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
        return path
    if not isinstance(source, Mapping):
        raise TypeError("A checkpoint must be a local path or a Hugging Face mapping")

    supported = {"repo_id", "filename", "revision", "cache_dir", "local_files_only", "sha256"}
    unknown = set(source) - supported
    if unknown:
        raise ValueError(f"Unknown checkpoint settings: {', '.join(sorted(unknown))}")
    repo_id, filename = source.get("repo_id"), source.get("filename")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("A Hugging Face checkpoint needs a repo_id")
    if (
        not isinstance(filename, str)
        or not filename.strip()
        or PurePosixPath(filename).is_absolute()
        or ".." in PurePosixPath(filename).parts
        or "\\" in filename
    ):
        raise ValueError("A Hugging Face checkpoint needs a relative filename")
    revision = source.get("revision", "main")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Checkpoint revision must be a nonempty string or null")
    local_only = source.get("local_files_only", False)
    if not isinstance(local_only, bool):
        raise ValueError("local_files_only must be true or false")
    expected = source.get("sha256")
    if expected is not None and (
        not isinstance(expected, str) or re.fullmatch(r"[a-fA-F0-9]{64}", expected) is None
    ):
        raise ValueError("Checkpoint sha256 must contain 64 hexadecimal characters")
    cache = source.get("cache_dir")
    if cache is not None:
        cache = Path(cache).expanduser()

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            cache_dir=cache,
            local_files_only=local_only,
        )
    )
    if expected:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError(f"Checkpoint SHA-256 mismatch: {repo_id}/{filename}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("overrides", nargs="*", help="YAML key=value overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    OmegaConf.resolve(config)
    sources = config.get("checkpoints")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("The configuration must define its checkpoints")
    for name, source in sources.items():
        print(f"{name}: {resolve_checkpoint(source)}")


if __name__ == "__main__":
    main()
