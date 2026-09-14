# ms-swift source for VPTracker

This directory contains the ms-swift source used by VPTracker for supervised
fine-tuning and inference, plus the VPTracker visual-prompt plugin.

- Upstream: [modelscope/ms-swift v4.5.3](https://github.com/modelscope/ms-swift/releases/tag/v4.5.3).
- Upstream commit: [`faed594ce0edabdd82a50a3d4111b0f0ffd9487a`](https://github.com/modelscope/ms-swift/commit/faed594ce0edabdd82a50a3d4111b0f0ffd9487a).
- Local package version: `4.5.3+vptracker1`.
- License: [Apache-2.0](LICENSE), with upstream copyright notices preserved.

## VPTracker changes

The source is trimmed to the training and inference components needed by this
project. Dedicated reinforcement-learning trainers, reward implementations,
rollout services, Megatron orchestration, and web UI components are removed.
Their entry points and imports are removed as well. Shared SFT/inference
utilities, model and template implementations, and framework configuration files
are retained. Upstream examples, tests, and documentation are omitted.

`swift/template/vptracker_plugin.py` registers the `vptracker` template and preserves each
training row's `visual_prompt` metadata. It uses `../vptracker/visual_prompt.py`
for template cropping and search-image annotation, keeping training and tracking
preprocessing consistent. `train.sh` loads this plugin explicitly.

## Install and run

From the VPTracker repository root:

```bash
python -m pip install -r requirements-train.txt
GPUS=1 bash train.sh
```

The requirements use `-e ./ms-swift`. The training script also puts this source
directory on `PYTHONPATH`, so it runs the bundled implementation. Dataset building
and frame-by-frame tracking are documented in the [project README](../README.md).
