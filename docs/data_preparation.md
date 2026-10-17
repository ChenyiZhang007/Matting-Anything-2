# Dataset preparation

[Back to README](../README.md#dataset-preparation)

Download the training splits below and arrange them under `<DATA_ROOT>` as shown.
Set `paths.data_root` in [configs/datasets.yaml](../configs/datasets.yaml) to this directory.

| Data | Author's download / project page |
| --- | --- |
| DIS5K, DIS-TR | [DIS](https://github.com/xuebinqin/DIS) |
| DUTS-TR | [DUTS](https://saliencydetection.net/duts/) |
| Adobe DIM / Composition-1K foregrounds and alphas | [Deep Image Matting](https://sites.google.com/view/deepimagematting) |
| Distinctions-646, training foregrounds and alphas | [HAttMatting](https://github.com/yuhaoliu7456/CVPR2020-HAttMatting) |
| AM-2K, training split | [GFM](https://github.com/JizhiziLi/GFM) |
| P3M-10k, face-blurred training split | [P3M](https://github.com/JizhiziLi/P3M) |
| Transparent-460, training split | [TransMatting](https://github.com/AceCHQ/TransMatting) |
| COCO 2017 training images | [COCO downloads](https://cocodataset.org/#download) |
| VideoMatting108, training split | [TCVOM](https://github.com/yunkezhang/TCVOM) |
| MOSE, training split | [MOSE](https://mose.video/) |

```text
<DATA_ROOT>/
├── DIS5K/DIS-TR/
│   ├── im/<name>.jpg
│   └── gt/<name>.png
├── DUTS-TR/
│   ├── DUTS-TR-Image/<name>.jpg
│   └── DUTS-TR-Mask/<name>.png
├── Combined_Dataset/Training_set/combined/
│   ├── fg/<name>.png
│   └── alpha/<name>.png
├── Distinctions-646/Train/
│   ├── FG/<name>.png
│   └── GT/<name>.png
├── AM-2K/train/
│   ├── original/<name>.jpg
│   └── mask/<name>.png
├── P3M-10k/train/
│   ├── blurred_image/<name>.jpg
│   └── mask/<name>.png
├── Transparent-460/Train/
│   ├── fg/<name>.png
│   └── alpha/<name>.png
├── train2017/<background>.jpg
├── VideoMatting108/
│   ├── FG/FG_done/<category>/<video>/
│   │   ├── img/00000.png
│   │   └── alpha/00000.png
│   ├── BG/BG_done2/<background_video>/00001.jpg
│   ├── frame_corr.json
│   ├── train_videos.txt
│   └── val_videos.txt
└── MOSE/train/
    ├── JPEGImages/<video>/00000.jpg
    └── Annotations/<video>/00000.png
```

## VideoMatting108 batch conversion

Extract VM108 RGBA PNGs to
`<SOURCE_ROOT>/FG/FG_done/<category>/<video>/` (or its `rgba/` subfolder), and
background frames to `<SOURCE_ROOT>/BG/BG_done2/`. Use our
[conversion script](../mam2/data/prepare_vm108.py) with a separate output directory:

```bash
mam2-prepare-vm108 --config configs/prepare_vm108.yaml \
  paths.source_root=/path/to/downloaded/VideoMatting108 \
  paths.output_root=/path/to/datasets/VideoMatting108
```

The script prepares RGB/alpha PNGs and the included `frame_corr.json` and split
lists in the layout above. Backgrounds are linked from the source directory.
