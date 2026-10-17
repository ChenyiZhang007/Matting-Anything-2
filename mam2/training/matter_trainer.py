"""Stage 2: optimize the lightweight matter on image-matting data only."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from mam2.checkpoints import read_model_state
from mam2.models.build import build_matter
from mam2.training.dataset.matting import ImageMattingDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_teacher_weights(teacher, checkpoint, *, resize_position_embeddings=True):
    """Load ViTMatte weights, adapting only relative-position table lengths."""
    complete = read_model_state(checkpoint)
    prefix = next(
        (
            candidate
            for candidate in ("teacher_backbone.", "backbone.")
            if any(name.startswith(candidate) for name in complete)
        ),
        "",
    )
    state = {
        name.removeprefix(prefix): value
        for name, value in complete.items()
        if not prefix or name.startswith(prefix)
    }
    expected = teacher.state_dict()
    if resize_position_embeddings:
        for name, value in state.items():
            target = expected.get(name)
            if (
                target is not None
                and name.endswith((".rel_pos_h", ".rel_pos_w"))
                and value.ndim == target.ndim == 2
                and value.shape[1] == target.shape[1]
                and value.shape[0] != target.shape[0]
            ):
                state[name] = F.interpolate(
                    value.T[None].float(),
                    size=target.shape[0],
                    mode="linear",
                    align_corners=False,
                )[0].T.to(value.dtype)
                logging.info("Resize teacher %s: %s -> %s", name, value.shape, target.shape)
    teacher.load_state_dict(state, strict=True)
    return state, complete


def train_matter(config):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_type = torch.device(config.runtime.device).type
    device = torch.device(device_type, local_rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    seed = int(config.runtime.seed) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    settings = config.training
    model = build_matter(config.matter, settings.init_checkpoint, device).train()
    criterion = instantiate(config.loss, _convert_="all").to(device)
    teacher = None
    if settings.distillation_weight:
        teacher = instantiate(config.teacher, _convert_="all").to(device).eval()
        state, complete_state = load_teacher_weights(
            teacher,
            settings.teacher_checkpoint,
            resize_position_embeddings=settings.get("resize_teacher_position_embeddings", True),
        )
        teacher.requires_grad_(False)
        if settings.init_checkpoint is None and settings.resume_checkpoint is None:
            # Appendix A.1: retain the teacher's backbone/detail weights and
            # initialize only token routing / refinement modules from scratch.
            initial = {f"backbone.{name}": value for name, value in state.items()}
            initial.update(
                {
                    name: value
                    for name, value in complete_state.items()
                    if name.startswith("decoder.")
                }
            )
            expected = model.state_dict()
            initial = {
                name: value
                for name, value in initial.items()
                if name in expected and expected[name].shape == value.shape
            }
            missing, _ = model.load_state_dict(initial, strict=False)
            unexplained = [
                name
                for name in missing
                if ".routers." not in name and ".efficient_block." not in name
            ]
            if unexplained:
                raise ValueError(
                    "Teacher initialization needs compatible backbone and decoder weights. "
                    "Supply a full ViTMatte checkpoint or training.init_checkpoint. "
                    f"Missing parameters: {unexplained}"
                )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        config.runtime.precision
    ]
    scaler = torch.amp.GradScaler(device_type, enabled=dtype == torch.float16)
    iteration = 0
    if settings.resume_checkpoint:
        checkpoint = torch.load(settings.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        iteration = checkpoint["iteration"]
    dataset = ImageMattingDataset(
        OmegaConf.to_container(config.data.matting_sources, resolve=True),
        config.paths.backgrounds,
        OmegaConf.to_container(config.augmentation, resolve=True),
    )
    sampler = DistributedSampler(dataset, seed=int(config.runtime.seed))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        sampler=sampler,
        num_workers=settings.workers,
        drop_last=True,
        pin_memory=device_type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("Training dataset is smaller than one distributed batch")
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device_type == "cuda" else None,
        find_unused_parameters=True,
    )
    output = Path(config.launcher.experiment_log_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    epoch = iteration // len(loader)
    while iteration < settings.iterations:
        sampler.set_epoch(epoch)
        for batch in loader:
            if iteration >= settings.iterations:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type, dtype=dtype, enabled=dtype != torch.float32):
                alpha, routing, student_features = model(
                    batch["image"], batch["trimap"], return_features=True
                )
                # Pixel losses use FP32 even when the backbone uses autocast.
                with torch.autocast(device_type, enabled=False):
                    losses = criterion(alpha.float(), batch["alpha"], batch["trimap"])
                if teacher is not None:
                    normalized = (batch["image"] - model.module.pixel_mean) / model.module.pixel_std
                    inputs = torch.cat((normalized, batch["trimap"]), dim=1)
                    height, width = inputs.shape[-2:]
                    multiple = model.module.size_divisibility
                    inputs = F.pad(inputs, (0, (-width) % multiple, 0, (-height) % multiple))
                    with torch.no_grad():
                        teacher_features = teacher(inputs)
                    losses["distillation"] = (
                        F.mse_loss(student_features.float(), teacher_features.float())
                        * settings.distillation_weight
                    )
                if routing and settings.routing_weight:
                    ratio = torch.stack([value.float().mean() for value in routing]).mean()
                    losses["routing"] = (
                        ratio - config.matter.backbone.topk
                    ).square() * settings.routing_weight
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite Stage 2 loss at iteration {iteration}")
            scaler.scale(total).backward()
            if settings.gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            if rank == 0 and iteration % settings.log_every == 0:
                logging.info(
                    "iteration=%d %s",
                    iteration,
                    {name: float(value.detach()) for name, value in losses.items()},
                )
            if rank == 0 and (
                iteration % settings.checkpoint_every == 0 or iteration == settings.iterations
            ):
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "iteration": iteration,
                }
                temporary = output / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "checkpoint.pt")
        epoch += 1
