"""Launch either training stage locally or under torchrun."""

import argparse
import os
import socket
from pathlib import Path

from omegaconf import OmegaConf

from mam2.config import load_config


def _run(rank, world_size, config):
    import torch.distributed as distributed
    from hydra.utils import instantiate

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    try:
        if config.stage == "main":
            instantiate(config.trainer, _recursive_=False).run()
        elif config.stage == "matter":
            from mam2.training.matter_trainer import train_matter

            train_matter(config)
        else:
            raise ValueError(f"Unknown training stage: {config.stage}")
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage 1 or Stage 2 YAML")
    parser.add_argument(
        "--validate-config", action="store_true", help="Resolve YAML without starting training"
    )
    parser.add_argument("overrides", nargs="*", help="YAML key=value overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    OmegaConf.resolve(config)
    # Inspect all mandatory values before starting any worker.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if args.validate_config:
        print(OmegaConf.to_yaml(config))
        return
    devices = config.runtime.visible_devices
    if devices is not None:
        if "LOCAL_RANK" in os.environ:
            raise ValueError(
                "Set CUDA_VISIBLE_DEVICES before torchrun instead of runtime.visible_devices"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    import torch.multiprocessing as multiprocessing

    output = Path(config.launcher.experiment_log_dir)
    output.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", 0)) == 0:
        OmegaConf.save(config, output / "config_resolved.yaml")
    if "LOCAL_RANK" in os.environ:
        _run(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), config)
        return
    os.environ["MASTER_ADDR"] = config.launcher.master_address
    port = config.launcher.master_port
    if port is None:
        with socket.socket() as sock:
            sock.bind((config.launcher.master_address, 0))
            port = sock.getsockname()[1]
    os.environ["MASTER_PORT"] = str(port)
    count = int(config.launcher.processes)
    if count < 1:
        raise ValueError("launcher.processes must be positive")
    if count == 1:
        _run(0, 1, config)
    else:
        multiprocessing.spawn(_run, args=(count, config), nprocs=count, join=True)


if __name__ == "__main__":
    main()
