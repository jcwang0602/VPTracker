#!/usr/bin/env python3
"""Time full-dataset VPTracker inference with one serial video sequence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "069": {
        "checkpoint": ROOT / (
            "outputs_vpt_r1/checkpoints/"
            "069_qwen35_2b_vp_blue_w1_op100_out25_pixelcoords_no_imgtok_cap_"
            "1m_lr2e-5_bs128_e1_pdbs4_ga4/v0-20260718-225749/checkpoint-7813"
        ),
        "model_type": "qwen3_5",
        "model": "Qwen3.5-2B full fine-tune",
        "coordinate_format": "pixel",
        "norm_coords": False,
    },
}
DEFAULT_TNL2K = Path("/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnl2k/test")
DEFAULT_TNLLT = Path("/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnllt")
DEFAULT_TNLLT_SPLIT = ROOT / "data_specs/tnllt_test_split.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="069")
    parser.add_argument("--dataset", required=True, choices=("tnl2k", "tnllt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tnl2k-root", type=Path, default=DEFAULT_TNL2K)
    parser.add_argument("--tnllt-root", type=Path, default=DEFAULT_TNLLT)
    parser.add_argument("--tnllt-split", type=Path, default=DEFAULT_TNLLT_SPLIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = EXPERIMENTS[args.experiment]["checkpoint"]
    return args


def dataset_names(args: argparse.Namespace) -> list[str]:
    if args.dataset == "tnl2k":
        names = sorted(name for name in os.listdir(args.tnl2k_root) if (args.tnl2k_root / name).is_dir())
    else:
        names = [line.strip() for line in args.tnllt_split.read_text().splitlines() if line.strip()]
    if args.max_sequences is not None:
        names = names[: args.max_sequences]
    return names


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    root = args.tnl2k_root if args.dataset == "tnl2k" else args.tnllt_root
    sequences = []
    for name in dataset_names(args):
        sequence_dir = root / name
        for required in (sequence_dir / "imgs", sequence_dir / "language.txt", sequence_dir / "groundtruth.txt"):
            if not required.exists():
                raise FileNotFoundError(required)
        frame_count = len(os.listdir(sequence_dir / "imgs"))
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)
        sequences.append({"name": name, "frame_count": frame_count})
    return {
        "dataset": args.dataset,
        "sequence_count": len(sequences),
        "frame_count": sum(item["frame_count"] for item in sequences),
        "sequences": sequences,
    }


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def build_summary(
    args: argparse.Namespace,
    expected: dict[str, Any],
    sequence_results: list[dict[str, Any]],
    engine_load_seconds: float,
    dataset_wall_seconds: float,
    end_to_end_seconds: float,
    complete: bool,
) -> dict[str, Any]:
    experiment = EXPERIMENTS[args.experiment]
    completed_frames = sum(item["frame_count"] for item in sequence_results)
    model_calls = sum(item["model_calls"] for item in sequence_results)
    inference_seconds = sum(item["inference_seconds"] for item in sequence_results)
    invalid = sum(item["invalid_response_count"] for item in sequence_results)
    return {
        "schema_version": 1,
        "complete": complete,
        "updated_at": utc_now(),
        "experiment": args.experiment,
        "dataset": args.dataset,
        "protocol": {
            "model": experiment["model"],
            "model_type": experiment["model_type"],
            "coordinate_format": experiment["coordinate_format"],
            "visual_prompt": True,
            "visual_prompt_color": "blue",
            "visual_prompt_width": 1,
            "visual_prompt_line_style": "solid",
            "visual_prompt_opacity": 1.0,
            "visual_prompt_scale": 4,
            "visual_prompt_placement": "previous_prediction",
            "template_scale": 2.0,
            "image_max_token_num": 16384,
            "max_output_tokens": 128,
            "temperature": 0.0,
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "vllm_gpu_memory_utilization": 0.9,
            "request_batch_size": 1,
            "sequence_parallelism": 1,
            "frame_processing": "serial",
            "gpu_count": 1,
        },
        "environment": {
            "gpu": gpu_name(),
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str((args.tnl2k_root if args.dataset == "tnl2k" else args.tnllt_root).resolve()),
            "tnllt_split": str(args.tnllt_split.resolve()) if args.dataset == "tnllt" else None,
        },
        "expected": {
            "sequence_count": expected["sequence_count"],
            "frame_count": expected["frame_count"],
        },
        "completed": {
            "sequence_count": len(sequence_results),
            "frame_count": completed_frames,
        },
        "timing": {
            "vllm_engine_cold_load_seconds": engine_load_seconds,
            "dataset_wall_seconds": dataset_wall_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "dataset_fps": completed_frames / dataset_wall_seconds if dataset_wall_seconds else None,
            "model_call_count": model_calls,
            "model_inference_seconds": inference_seconds,
            "mean_model_call_seconds": inference_seconds / model_calls if model_calls else None,
            "invalid_response_count": invalid,
        },
        "sequences": sequence_results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment = EXPERIMENTS[args.experiment]
    expected = inventory(args)
    summary_path = args.output_dir / "summary.json"
    if args.inventory_only:
        atomic_json(summary_path, expected)
        return expected

    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Invalid checkpoint: {args.checkpoint}")

    import torch
    from swift import get_model_processor, get_template
    from swift.infer_engine import RequestConfig, VllmEngine
    from evaluation.infer_tracking_qwen_vlt import (
        build_infer_request,
        make_sequence_state,
        parse_response,
        save_sequence,
    )
    from evaluation.tnl2k_dataset import TNL2KDataset
    from evaluation.tnllt_dataset import TNLLTDataset

    runtime_args = argparse.Namespace(
        visual_prompt=True,
        balanced_vp_prompt=False,
        balanced_prompt=False,
        compact_prompt=False,
        vp_scale=4,
        vp_color="blue",
        vp_width=1,
        vp_line_style="solid",
        vp_opacity=1.0,
        vp_placement="previous_prediction",
        vp_random_seed=0,
        template_scale=2.0,
        save_image=False,
        norm_coords=experiment["norm_coords"],
    )
    if args.dataset == "tnl2k":
        dataset = TNL2KDataset(root_dir=str(args.tnl2k_root), seg_index=0, seg_total=1, debug=False)
    else:
        dataset = TNLLTDataset(
            root_dir=str(args.tnllt_root), split_path=str(args.tnllt_split),
            seg_index=0, seg_total=1, debug=False,
        )
    if args.max_sequences is not None:
        dataset.video_names = dataset.video_names[: args.max_sequences]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = args.output_dir / "tracking_results"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    request_config = RequestConfig(max_tokens=128, temperature=0)

    job_started = time.perf_counter()
    load_started = time.perf_counter()
    _, processor = get_model_processor(
        str(args.checkpoint), load_model=False, model_type=experiment["model_type"]
    )
    template = get_template(processor, enable_thinking=False)
    engine = VllmEngine(
        str(args.checkpoint),
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_num_seqs=1,
        torch_dtype=torch.bfloat16,
        model_type=experiment["model_type"],
        template=template,
    )
    engine_load_seconds = time.perf_counter() - load_started

    sequence_results: list[dict[str, Any]] = []
    dataset_started = time.perf_counter()
    for index in range(len(dataset)):
        sequence_started = time.perf_counter()
        sample = dataset[index]
        if args.max_frames is not None:
            sample["image_paths"] = sample["image_paths"][: args.max_frames]
            sample["bboxes"] = sample["bboxes"][: args.max_frames]
        state = make_sequence_state(sample, str(tracking_dir))
        inference_seconds = 0.0
        request_seconds = 0.0
        parse_seconds = 0.0
        invalid_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        usage_records = 0

        while state["frame_index"] < len(state["image_paths"]):
            request_started = time.perf_counter()
            request = build_infer_request(runtime_args, state)
            request_seconds += time.perf_counter() - request_started

            infer_started = time.perf_counter()
            response_object = engine.infer([request], request_config)[0]
            inference_seconds += time.perf_counter() - infer_started
            response = response_object.choices[0].message.content

            parse_started = time.perf_counter()
            bbox, success = parse_response(
                response, state["img_width"], state["img_height"],
                experiment["norm_coords"],
            )
            if success:
                state["current_bbox"] = bbox
            else:
                invalid_count += 1
            state["result_bbox"].append(state["current_bbox"])
            state["frame_index"] += 1
            parse_seconds += time.perf_counter() - parse_started

            input_tokens, output_tokens = usage_tokens(response_object)
            if input_tokens is not None and output_tokens is not None:
                prompt_tokens += input_tokens
                completion_tokens += output_tokens
                usage_records += 1

        save_sequence(state, str(tracking_dir))
        elapsed = time.perf_counter() - sequence_started
        frame_count = len(state["image_paths"])
        result = {
            "index": index,
            "name": state["video_name"],
            "frame_count": frame_count,
            "model_calls": max(frame_count - 1, 0),
            "wall_seconds": elapsed,
            "fps": frame_count / elapsed,
            "inference_seconds": inference_seconds,
            "request_build_seconds": request_seconds,
            "response_parse_seconds": parse_seconds,
            "invalid_response_count": invalid_count,
            "prompt_tokens": prompt_tokens if usage_records else None,
            "completion_tokens": completion_tokens if usage_records else None,
        }
        sequence_results.append(result)
        partial = build_summary(
            args, expected, sequence_results, engine_load_seconds,
            time.perf_counter() - dataset_started,
            time.perf_counter() - job_started, complete=False,
        )
        atomic_json(summary_path, partial)
        print(
            f"[{index + 1}/{len(dataset)}] {state['video_name']}: "
            f"{frame_count} frames, {elapsed:.3f}s, {frame_count / elapsed:.3f} FPS, "
            f"invalid={invalid_count}",
            flush=True,
        )

    dataset_wall_seconds = time.perf_counter() - dataset_started
    result = build_summary(
        args, expected, sequence_results, engine_load_seconds,
        dataset_wall_seconds, time.perf_counter() - job_started, complete=True,
    )
    atomic_json(summary_path, result)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "complete": result.get("complete"),
        "dataset": result["dataset"],
        "sequence_count": result.get("completed", {}).get("sequence_count", result.get("sequence_count")),
        "frame_count": result.get("completed", {}).get("frame_count", result.get("frame_count")),
        "timing": result.get("timing"),
    }, indent=2))


if __name__ == "__main__":
    main()
