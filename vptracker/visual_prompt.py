"""Template cropping and search-image prompts, shared by SFT and inference.

Adapted from VPTracker's original Apache-2.0 ms-swift customization. Images are
prepared before the model processor resizes them; output boxes stay in the
coordinate system of the full search image.
"""

import math
import random

from PIL import ImageDraw


def crop_template(image, bbox, scale=2.0):
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(v) for v in bbox) or x2 <= x1 or y2 <= y1 or scale <= 0:
        raise ValueError("The template requires a finite, nonempty xyxy box and positive scale")
    side = math.ceil(math.sqrt((x2 - x1) * (y2 - y1)) * scale)
    left = round((x1 + x2 - side) / 2)
    top = round((y1 + y2 - side) / 2)
    # Keep the original crop convention, including its right/bottom border limit.
    bounds = (max(0, left), max(0, top), min(left + side, image.width - 1), min(top + side, image.height - 1))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("The template box does not intersect the image")
    return image.crop(bounds)


def draw_search_prompt(image, bbox, scale=3, *, phase="test", inbbox_ratio=0.75,
                       color="blue", width=1, rng=None):
    rng = rng or random
    if phase not in {"train", "test"} or scale <= 0 or width < 1 or not 0 <= inbbox_ratio <= 1:
        raise ValueError("Invalid visual-prompt settings")
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Expected four finite pixel coordinates")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 0 or h < 0:
        raise ValueError("Visual-prompt xyxy coordinates are reversed")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = w * scale, h * scale
    if phase == "train":
        min_dx, min_dy = (new_w - w) / 2, (new_h - h) / 2
        if rng.random() < inbbox_ratio:
            dx, dy = rng.uniform(-min_dx, min_dx), rng.uniform(-min_dy, min_dy)
        else:
            # Retain the training displacement distribution. After clipping,
            # this branch does not guarantee a non-overlapping prompt box.
            max_dx, max_dy = image.width - new_w - min_dx, image.height - new_h - min_dy
            dx = rng.choice([rng.uniform(min_dx, max_dx), rng.uniform(-max_dx, -min_dx)])
            dy = rng.choice([rng.uniform(min_dy, max_dy), rng.uniform(-max_dy, -min_dy)])
        cx, cy = cx + dx, cy + dy
    box = [max(0, int(cx - new_w / 2)), max(0, int(cy - new_h / 2)),
           min(image.width, int(cx + new_w / 2)), min(image.height, int(cy + new_h / 2))]
    if box[2] < box[0] or box[3] < box[1]:
        # The original renderer used a random valid box for out-of-image shifts.
        box[0], box[2] = sorted(rng.randint(1, max(1, image.width - 1)) for _ in range(2))
        box[1], box[3] = sorted(rng.randint(1, max(1, image.height - 1)) for _ in range(2))
    result = image.copy()
    ImageDraw.Draw(result).rectangle(box, outline=color, width=width)
    return result


def prepare_tracking_images(template_image, search_image, template_bbox, search_bbox,
                            vp_scale=3, template_scale=2.0, *, phase="test", inbbox_ratio=0.75,
                            color="blue", width=1, rng=None):
    template = crop_template(template_image.convert("RGB"), template_bbox, template_scale)
    search = draw_search_prompt(search_image.convert("RGB"), search_bbox, vp_scale,
                                phase=phase, inbbox_ratio=inbbox_ratio, color=color, width=width, rng=rng)
    return template, search
