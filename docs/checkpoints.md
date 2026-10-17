# Hugging Face checkpoints

[Back to README](../README.md#quick-installation)

The public model repository is
[Chenyi007/Matting-Anything-2](https://huggingface.co/Chenyi007/Matting-Anything-2).
It contains the original experimental `checkpoint_100.pt` as `mam2_main.pt` and
the associated MEMatte checkpoint as `mam2_matter.pth`. The published files retain
their original bytes; `weights.json` and `SHA256SUMS` record their sizes and hashes.

## Download and cache

After installing the source project, download both weights with:

```bash
mam2-download --config configs/inference.yaml
```

The demo and inference CLI use the same resolver, so this step is optional.
Without prefetching, files download when the models first load. Downloading the
public weights does not require authentication.

## YAML configuration

[configs/inference.yaml](../configs/inference.yaml) defines the shared Hub
settings, inherited by [configs/web.yaml](../configs/web.yaml):

```yaml
huggingface:
  repo_id: Chenyi007/Matting-Anything-2
  revision: 0e0a8d0d9034b00c4baca88f1a0bb1397e724905
  cache_dir: null
  local_files_only: false
```

Each entry under `checkpoints` provides `repo_id`, `filename`, `revision`,
`cache_dir`, `local_files_only` and an optional `sha256`. The default main and
matter entries reference the shared settings above. Downloads use the standard
Hugging Face cache; set `cache_dir` to choose another location:

```bash
mam2-download --config configs/inference.yaml \
  huggingface.cache_dir=/path/to/mam2-cache
```

The release pins an immutable commit containing both weights and validates
SHA-256 before loading. To publish or use different weights, update their
filenames, revision and hashes together. Changing the repository alone does
not change the expected file contents.

## Offline use

Once both files have been downloaded, use the same cache directory and revision
with `local_files_only=true`:

```bash
CUDA_VISIBLE_DEVICES=0 mam2-demo --config configs/web.yaml \
  huggingface.cache_dir=/path/to/mam2-cache \
  huggingface.local_files_only=true
```

Offline mode requires the configured files to be present in that cache already.

## Local files

A checkpoint entry can also be a local filename. This supports existing
checkpoints and setups that manage downloads separately:

```bash
mam2-demo --config configs/web.yaml \
  checkpoints.main=/path/to/main_checkpoint.pt \
  checkpoints.matter=/path/to/MEMatte.pth
```

The same overrides apply to `mam2-infer`. Local paths are expanded with `~` and
loaded directly. Historical main-model keys named `sam_mask_decoder.*` are
migrated to `pdd.*`; model state is loaded strictly.

For private repositories, authenticate with `hf auth login` or supply `HF_TOKEN`
through your environment. Keep credentials outside YAML and source files.
