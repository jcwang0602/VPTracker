"""Build the TNL2K/TNLLT supervised JSONL consumed by the VPTracker Swift plugin."""

import argparse
import json
import os
from pathlib import Path
import random
import tempfile

from dataset.loaders import TNLLT_TRAIN_SPLIT, Sequence, load_training_sequences
from vptracker.prompts import build_tracking_prompt


def sample_record(sequence: Sequence, rng: random.Random, *, template_scale: float = 2.0,
                  color: str = "blue", width: int = 1, inbbox_ratio: float = 0.75) -> dict:
    """Sample a visible template followed by a possibly absent search frame."""
    template_index = rng.choice(sequence.template_indices)
    search_index = rng.randrange(template_index + 1, len(sequence.frames))
    template_bbox = sequence.bbox(template_index)
    search_bbox = sequence.bbox(search_index)
    answer = {"visible": "yes" if sequence.visible[search_index] else "no", "bbox": search_bbox}
    return {
        "messages": [
            {"role": "user", "content": build_tracking_prompt(sequence.language, color=color)},
            {"role": "assistant", "content": "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"},
        ],
        "images": [sequence.image_path(template_index), sequence.image_path(search_index)],
        "visual_prompt": {
            "phase": "train", "enable": True, "scale": rng.randint(2, 8),
            "color": color, "width": width, "inbbox_ratio": inbbox_ratio,
            "template_bbox": template_bbox, "template_scale": template_scale,
            "search_bbox": search_bbox, "line_style": "solid", "opacity": 1.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tnl2k-root", type=Path, required=True, help="TNL2K root containing train/")
    parser.add_argument("--tnllt-root", type=Path, required=True, help="TNLLT root containing sequence directories")
    parser.add_argument("--tnllt-split", type=Path, default=TNLLT_TRAIN_SPLIT,
                        help="Published TNLLT training list or a subset (never a test split)")
    parser.add_argument("--output", type=Path, required=True, help="Output training JSONL")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tnl2k-ratio", type=float, default=0.7, help="Fraction of samples from TNL2K")
    parser.add_argument("--template-scale", type=float, default=2.0)
    parser.add_argument("--vp-color", default="blue")
    parser.add_argument("--vp-width", type=int, default=1)
    parser.add_argument("--inbbox-ratio", type=float, default=0.75)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not 0 <= args.tnl2k_ratio <= 1 or not 0 <= args.inbbox_ratio <= 1:
        parser.error("--tnl2k-ratio and --inbbox-ratio must be between 0 and 1")
    if args.template_scale <= 0 or args.vp_width < 1:
        parser.error("--template-scale and --vp-width must be positive")
    return args


def main() -> None:
    args = parse_args()
    datasets = load_training_sequences(args.tnl2k_root, args.tnllt_root, args.tnllt_split)
    rng = random.Random(args.seed)
    tnl2k_count = round(args.samples * args.tnl2k_ratio)
    remaining = [tnl2k_count, args.samples - tnl2k_count]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for index in range(args.samples):
                # Random ordering with exact source counts, without holding the JSONL in memory.
                source = 0 if rng.randrange(args.samples - index) < remaining[0] else 1
                remaining[source] -= 1
                record = sample_record(rng.choice(datasets[source]), rng,
                                       template_scale=args.template_scale, color=args.vp_color,
                                       width=args.vp_width, inbbox_ratio=args.inbbox_ratio)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if (index + 1) % 10_000 == 0:
                    print(f"Wrote {index + 1:,}/{args.samples:,} samples", flush=True)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Saved {output}: {args.samples:,} samples (TNL2K {tnl2k_count:,}, "
          f"TNLLT {args.samples - tnl2k_count:,}); seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
