# VPTracker

VPTracker is a global vision-language tracker that uses location-aware visual prompts with a multimodal language model. The current released checkpoint is experiment **069**, a full-parameter fine-tune of **Qwen3.5-2B**.

This repository contains dataset conversion, training launchers, inference, evaluation, and reward plugins. Large datasets, checkpoints, and generated results are excluded from Git. The 069 model is published at [jcwang0602/VPTracker](https://huggingface.co/jcwang0602/VPTracker).

## Installation

Use Python 3.10 or newer with a CUDA-compatible PyTorch build. Install the exact upstream ms-swift release recorded in [`third_party/ms_swift.lock.json`](third_party/ms_swift.lock.json), then project dependencies:

```bash
git clone --branch v4.5.3 --depth 1 https://github.com/modelscope/ms-swift.git
python -m pip install -e ms-swift
python -m pip install -r requirements-vptrack.txt
```

Do not install a second, unpinned copy of ms-swift after the editable install. Qwen3.5 loading requires a recent Transformers build; the old `transformers<4.55` constraint has been removed.

## Model and data paths

Download the checkpoint from Hugging Face, or point scripts at a local checkpoint. Training and evaluation scripts use the repository root by default and accept explicit dataset/checkpoint arguments where applicable. Dataset directories (`data/`), model directories (`models/`), and generated outputs remain ignored by `.gitignore`.

The released checkpoint was trained with seed 42 and records ms-swift `4.4.0.dev0` in its training metadata. The repository records upstream ms-swift v4.5.3 for maintenance and future runs; perform a loading smoke test before reproducing old metrics.

## Quick inference

```bash
VPTRACK_MODEL=jcwang0602/VPTracker python demo_vllm.py
```

For dataset evaluation, see `evaluation/infer_tracking_qwen_vlt.py` and scripts under `evaluation/`. For training launchers, see `train_scripts/`.

## License and citation

VPTracker code is released under the license in [`LICENSE`](LICENSE). ms-swift is an independent Apache-2.0 project; its source and notices are obtained from the upstream release linked above.

```bibtex
@misc{wang2025vptrackerglobalvisionlanguagetracking,
  title={VPTracker: Global Vision-Language Tracking via Visual Prompt and MLLM},
  author={Jingchao Wang and Kaiwen Zhou and Zhijian Wu and Kunhua Ji and Dingjiang Huang and Yefeng Zheng},
  year={2025},
  eprint={2512.22799},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
