---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - vision-language
  - video
  - tracking
base_model:
  - Qwen/Qwen3.5-2B-Base
---

# VPTracker (Qwen3.5-2B)

This repository contains the VPTracker full-parameter fine-tune from experiment 069. It uses **Qwen3.5-2B** as the base model and location-aware visual prompts for global vision-language tracking.

The checkpoint was trained with seed 42 and the ms-swift 4.4.0 development environment. The current code records upstream ms-swift v4.5.3 for inference and future training; verify the loading smoke test after upgrading runtime dependencies.

## Loading

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_id = "jcwang0602/VPTracker"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
```

The model card describes the 069 checkpoint only. Dataset preparation, visual-prompt rendering, evaluation, and reward plugins are available in the [VPTracker code repository](https://github.com/jcwang0602/VPTracker).
