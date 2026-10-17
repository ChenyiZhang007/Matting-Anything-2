# Third-party components

This release was extracted from the local Matting-Anything-2 research repository.
Original copyright notices are retained in copied source files.

| Component | Upstream | Use in this project |
| --- | --- | --- |
| SAM2 | https://github.com/facebookresearch/sam2 | Image backbone, prompts, memory and tracking; Apache 2.0, included `LICENSE` |
| MEMatte | https://github.com/linyiheng123/MEMatte | Lightweight matting backbone and detail decoder |
| ViTMatte | https://github.com/hustvl/ViTMatte | MEMatte's matting backbone/decoder lineage |
| GCA-Matting | https://github.com/Yaoyi-Li/GCA-Matting | Image-matting augmentations used by MEMatte |
| MatAnyone2 / Robust Video Matting | https://github.com/pq-yang/MatAnyone2, https://github.com/PeterL1n/RobustVideoMatting | Alpha metric definitions; reimplemented in PyTorch |
| Detectron2 / ViTDet | https://github.com/facebookresearch/detectron2 | Backbone and residual-block lineage; runtime layers use PyTorch and shared SAM2 primitives |

Model weights are hosted separately on
[Hugging Face](https://huggingface.co/Chenyi007/Matting-Anything-2). Dataset files,
third-party binaries and pretrained teacher checkpoints are not bundled with the
source. This document records source provenance; it does not replace the licenses
or attribution requirements of those upstream projects.
