"""Explicit image-pair sources for segmentation and matting training."""

from pathlib import Path

from mam2.training.dataset.vos_raw_dataset import VOSRawDataset

SUFFIXES = {".png", ".jpg", ".jpeg"}


def paired_images(sources):
    """Match by stem; source order, optional split lists and limits live in YAML."""
    pairs = []
    for source in sources:
        images = {
            p.stem: p for p in Path(source["images"]).iterdir() if p.suffix.lower() in SUFFIXES
        }
        alphas = {
            p.stem: p for p in Path(source["alphas"]).iterdir() if p.suffix.lower() in SUFFIXES
        }
        if source.get("file_list"):
            names = [
                Path(line.strip()).stem
                for line in Path(source["file_list"]).read_text().splitlines()
                if line.strip()
            ]
        else:
            names = sorted(images)
        excluded = set(source.get("exclude", []))
        names = [name for name in names if name not in excluded]
        if source.get("limit") is not None:
            names = names[: int(source["limit"])]
        for name in names:
            if name not in images or name not in alphas:
                raise FileNotFoundError(f"Missing image/alpha pair {name} in {source}")
            pairs.append((str(images[name]), str(alphas[name])))
    if not pairs:
        raise ValueError("No training image pairs were configured")
    return pairs


class ImageSegmentationRawDataset(VOSRawDataset):
    def __init__(self, sources):
        self.pairs = paired_images(sources)

    def get_sample(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class ImageMattingRawDataset(ImageSegmentationRawDataset):
    def __init__(self, sources, background_folder):
        super().__init__(sources)
        self.composition = []
        for source in sources:
            self.composition.extend(
                [
                    {
                        "probability": source.get("composite_probability", 1.0),
                        "scale_range": source.get("scale_range", [0.5, 1.0]),
                    }
                ]
                * len(paired_images([source]))
            )
        self.backgrounds = sorted(
            str(p) for p in Path(background_folder).iterdir() if p.suffix.lower() in SUFFIXES
        )
        if not self.backgrounds:
            raise ValueError(f"No training backgrounds in {background_folder}")

    def get_sample(self, index):
        import random

        return (*self.pairs[index], random.choice(self.backgrounds), self.composition[index])
