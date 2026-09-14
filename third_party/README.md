# Third-party components

VPTracker uses the upstream [ms-swift](https://github.com/modelscope/ms-swift) runtime. The release used for the Qwen3.5 code path is recorded in [`ms_swift.lock.json`](ms_swift.lock.json). Install the vendored source instead of installing a second copy from an unpinned Git URL:

```bash
python -m pip install -e ms-swift
```

Project-specific reward functions remain in `reward_funcs/` and are loaded with ms-swift's external plugin mechanism. Upstream source and license notices are intentionally kept separate from VPTracker code.
