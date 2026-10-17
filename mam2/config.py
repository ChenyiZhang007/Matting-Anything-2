"""Compose YAML files and apply command-line key=value overrides."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

for _name, _resolver in {
    "times": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "package": lambda relative: str(Path(__file__).parent / relative),
    "device_type": lambda device: str(device).split(":")[0],
    "amp_enabled": lambda precision: precision != "float32",
}.items():
    if not OmegaConf.has_resolver(_name):
        OmegaConf.register_new_resolver(_name, _resolver)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Resolve ``defaults`` relative to each YAML file, without changing cwd."""

    def compose(filename: Path, stack: tuple[Path, ...]) -> DictConfig:
        filename = filename.resolve()
        if filename in stack:
            raise ValueError(f"Circular YAML defaults: {filename}")
        current = OmegaConf.load(filename)
        if not isinstance(current, DictConfig):
            raise ValueError(f"Expected a YAML mapping: {filename}")
        defaults = current.pop("defaults", [])
        result = OmegaConf.create({})
        for relative in defaults:
            result = OmegaConf.merge(
                result, compose(filename.parent / relative, (*stack, filename))
            )
        return OmegaConf.merge(result, current)

    config = compose(Path(path), ())
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config
