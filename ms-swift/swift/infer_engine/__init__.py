# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for VPTracker: retain the SFT and inference source subset.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .base import BaseInferEngine
    from .infer_client import InferClient
    from .infer_engine import InferEngine
    from .lmdeploy_engine import LmdeployEngine
    from .protocol import ChatCompletionResponse, Function, InferRequest, RequestConfig
    from .sglang_engine import SglangEngine
    from .transformers_engine import TransformersEngine
    from .utils import AdapterRequest, patch_vllm_memory_leak, prepare_generation_config
    from .vllm_engine import VllmEngine
else:
    _import_structure = {
        'vllm_engine': ['VllmEngine'],
        'lmdeploy_engine': ['LmdeployEngine'],
        'sglang_engine': ['SglangEngine'],
        'transformers_engine': ['TransformersEngine'],
        'infer_client': ['InferClient'],
        'infer_engine': ['InferEngine'],
        'base': ['BaseInferEngine'],
        'utils': ['prepare_generation_config', 'AdapterRequest', 'patch_vllm_memory_leak'],
        'protocol': ['InferRequest', 'RequestConfig', 'Function', 'ChatCompletionResponse'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
