# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for VPTracker: retain the SFT and inference source subset.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .base import SwiftPipeline
    from .export import merge_lora
    from .infer import SwiftInfer, infer_main
    from .train import SwiftSft, sft_main
    from .utils import prepare_model_template
else:
    import sys

    _import_structure = {
        'infer': ['infer_main', 'SwiftInfer'],
        'export': ['merge_lora'],
        'train': ['sft_main', 'SwiftSft'],
        'base': ['SwiftPipeline'],
        'utils': ['prepare_model_template'],
    }
    sys.modules[__name__] = _LazyModule(
        __name__, globals()['__file__'], _import_structure, module_spec=__spec__, extra_objects={})
