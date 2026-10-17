"""Shared LoRA construction and explicit checkpoint compatibility."""

from collections.abc import Mapping
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from mam2.weights import resolve_checkpoint


def canonical_state_dict(state: Mapping) -> dict:
    """Accept historical SAM2 decoder names without changing tensor values."""
    result = {}
    for key, value in state.items():
        name = key.removeprefix("module.").replace("sam_mask_decoder.", "pdd.")
        if name in result:
            raise ValueError(f"Duplicate checkpoint key after migration: {name}")
        result[name] = value
    return result


def read_model_state(path: str | Path | Mapping) -> dict:
    checkpoint = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")
    return canonical_state_dict(state)


def load_weights(model: torch.nn.Module, path: str | Path | Mapping) -> None:
    """Require every inference weight to match; never silently skip parameters."""
    model.load_state_dict(read_model_state(path), strict=True)


def attach_lora(model: torch.nn.Module, config: Mapping | None) -> None:
    """Use the same adapter settings for training and inference."""
    if config is None or config.get("rank", 0) == 0:
        return
    rank = int(config["rank"])
    adapter = LoraConfig(
        r=rank,
        lora_alpha=config.get("alpha", rank),
        lora_dropout=config.get("dropout", 0.1),
        target_modules=list(config.get("target_modules", ["qkv", "proj"])),
        init_lora_weights="gaussian",
        use_rslora=False,
        use_dora=False,
    )
    model.image_encoder.trunk = get_peft_model(model.image_encoder.trunk, adapter)


def initialize_main(model, checkpoint_path, allowed_missing):
    """Initialize SAM2 parameters while leaving declared MAM2 additions trainable."""
    from fnmatch import fnmatchcase

    missing, unexpected = model.load_state_dict(read_model_state(checkpoint_path), strict=False)
    unexplained = [
        key for key in missing if not any(fnmatchcase(key, pattern) for pattern in allowed_missing)
    ]
    if unexplained or unexpected:
        raise RuntimeError(
            f"Pretrained model mismatch: missing={unexplained}, unexpected={unexpected}"
        )
    return model
