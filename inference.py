"""Track one frame directory with the public VPTracker Transformers model."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from PIL import Image

from vptracker.prompts import build_tracking_prompt
from vptracker.visual_prompt import prepare_tracking_images


def list_frames(directory):
    """Return image files in natural order (frame2 before frame10)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Frame directory does not exist: {directory}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    frames = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    frames.sort(key=lambda p: (
        [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)], p.name
    ))
    if not frames:
        raise ValueError(f"No image frames found in {directory}")
    return frames


def parse_box(response, image_size):
    """Read an absolute-pixel xyxy box; invalid or invisible answers return None."""
    answer = response.split("</think>")[-1].strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(answer)
        if isinstance(result, dict):
            if str(result.get("visible", "yes")).lower() in {"no", "false", "0"}:
                return None
            result = result.get("bbox")
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, list) or len(result) != 4:
            return None
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in result):
            return None
        x1, y1, x2, y2 = map(float, result)
        width, height = image_size
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
        return [x1, y1, x2, y2]
    except (ValueError, TypeError):
        return None


def tracking_messages(language, template_image, search_image):
    """Place the two images at the trained prompt's exact image positions."""
    parts = build_tracking_prompt(language, color="blue").split("<image>")
    if len(parts) != 3:
        raise ValueError("The tracking prompt must contain exactly two image placeholders")
    content = [{"type": "text", "text": parts[0]}]
    for image, text in zip((template_image, search_image), parts[1:]):
        content.extend([{"type": "image", "image": image}, {"type": "text", "text": text}])
    return [{"role": "user", "content": content}]


class VPTracker:
    def __init__(self, model="jcwang0602/VPTracker", device="auto", max_new_tokens=128):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model, dtype="auto", device_map=device, attn_implementation="sdpa"
        ).eval()
        self.max_new_tokens = max_new_tokens

    def predict(self, template_image, search_image, template_bbox, previous_bbox, language):
        """Predict xyxy coordinates in the original full search image."""
        template, search = prepare_tracking_images(
            template_image, search_image, template_bbox, previous_bbox,
            vp_scale=3, template_scale=2.0, phase="test", color="blue", width=1,
        )
        inputs = self.processor.apply_chat_template(
            tracking_messages(language, template, search),
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, use_cache=True,
            )
        response = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return parse_box(response, search_image.size)


def track_frames(model, frames, language, init_bbox, output, device="auto", max_new_tokens=128):
    """Write one x,y,width,height row per frame, including the initial frame."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("--language must contain a target description")
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    paths = list_frames(frames)
    with Image.open(paths[0]) as image:
        template = image.convert("RGB")
    x, y, width, height = init_bbox
    template_box = [x, y, x + width, y + height]
    if parse_box(json.dumps(template_box), template.size) is None:
        raise ValueError("--init-bbox must be a nonempty x y width height box inside the first frame")
    previous_box = list(template_box)
    tracker = VPTracker(model, device, max_new_tokens) if len(paths) > 1 else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as target:
            temporary = Path(target.name)
            for index, path in enumerate(paths):
                if index:
                    with Image.open(path) as image:
                        search = image.convert("RGB")
                    box = tracker.predict(template, search, template_box, previous_box, language)
                    if box is not None:
                        previous_box = box
                    else:
                        print(f"{path.name}: no valid visible box; keeping the previous position", file=sys.stderr)
                x1, y1, x2, y2 = previous_box
                target.write(",".join(f"{v:g}" for v in (x1, y1, x2 - x1, y2 - y1)) + "\n")
                print(f"[{index + 1}/{len(paths)}] {path.name}", file=sys.stderr)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jcwang0602/VPTracker", help="Hugging Face model ID or local model directory")
    parser.add_argument("--frames", required=True, type=Path, help="Directory of image frames, sorted naturally")
    parser.add_argument("--language", required=True, help="Description of the target object")
    parser.add_argument("--init-bbox", required=True, type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--output", required=True, type=Path, help="Output text file with one pixel-space xywh row per frame")
    parser.add_argument("--device", default="auto", help="Model device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    result = track_frames(**vars(args))
    print(f"Saved tracking results to {result}")


if __name__ == "__main__":
    main()
