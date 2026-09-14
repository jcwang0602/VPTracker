"""Register VPTracker's visual-prompt SFT template with ms-swift 4.5.3.

Loaded by train.sh through --external_plugins with the bundled Swift source.
Ordinary qwen3_5 samples keep their original template.
"""

from copy import deepcopy

from swift.dataset.preprocessor.core import RowPreprocessor
from swift.template import TEMPLATE_MAPPING, register_template
from swift.template.templates.qwen import Qwen3_5Template

from vptracker.visual_prompt import prepare_tracking_images


class VPTrackerTemplate(Qwen3_5Template):
    special_keys = Qwen3_5Template.special_keys + ["visual_prompt"]

    def _preprocess_inputs(self, inputs):
        vp = inputs.extra_kwargs.pop("visual_prompt", None)
        if not vp or not vp.get("enable") or len(inputs.images) != 2:
            raise ValueError("VPTracker SFT requires two images and visual_prompt metadata from dataset.build")
        if vp.get("line_style", "solid") != "solid" or vp.get("opacity", 1.0) != 1.0:
            raise ValueError("VPTracker uses an opaque solid visual prompt")
        template, search = [self._load_image(image, True) for image in inputs.images]
        inputs.images = list(prepare_tracking_images(
            template, search, vp["template_bbox"], vp["search_bbox"],
            vp_scale=vp["scale"], template_scale=vp["template_scale"],
            phase=vp["phase"], inbbox_ratio=vp["inbbox_ratio"],
            color=vp["color"], width=vp["width"],
        ))
        super()._preprocess_inputs(inputs)


if "visual_prompt" not in RowPreprocessor.standard_keys:
    RowPreprocessor.standard_keys.append("visual_prompt")
meta = deepcopy(TEMPLATE_MAPPING["qwen3_5"])
meta.template_type = "vptracker"
meta.template_cls = VPTrackerTemplate
register_template(meta, exist_ok=True)
