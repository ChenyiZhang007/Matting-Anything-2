"""Construct the main model and lightweight matter from resolved YAML nodes."""

from hydra.utils import instantiate
from omegaconf import OmegaConf

from mam2.checkpoints import attach_lora, load_weights


def build_predictor(config, checkpoint, lora, device):
    model = instantiate(config, _convert_="all")
    attach_lora(model, OmegaConf.to_container(lora, resolve=True))
    load_weights(model, checkpoint)
    return model.to(device).eval()


def build_matter(config, checkpoint, device):
    model = instantiate(config, _convert_="all")
    if checkpoint:
        load_weights(model, checkpoint)
    return model.to(device)
