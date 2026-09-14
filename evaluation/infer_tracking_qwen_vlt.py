import json
import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
from swift.infer_engine import VllmEngine, RequestConfig, InferRequest
from evaluation.tnl2k_dataset import TNL2KDataset
from evaluation.otb_dataset import OTBDataset
from evaluation.tnllt_dataset import TNLLTDataset
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bbox_utils import convert_x1y1x2y2_to_xywh, denormalize_coordinates
from utils.vis_utils import save_tracking_results
from swift import get_model_processor, get_template
from evaluation.prompts import (
    PROMPT_VLT,
    PROMPT_VLT_BALANCED,
    PROMPT_VLT_COMPACT,
    PROMPT_VLT_VP,
    PROMPT_VLT_VP_BALANCED,
    PROMPT_VLT_VP_COMPACT,
)
from evaluation.prompt_placement import random_size_matched_bbox


def check_bbox(bbox, img_width, img_height):
    # 有效
    valid_bbox = True
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
        valid_bbox = False
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        valid_bbox = False
    if (
        bbox[0] > img_width
        or bbox[1] > img_height
        or bbox[2] > img_width
        or bbox[3] > img_height
    ):
        valid_bbox = False
    return valid_bbox


def build_infer_request(args, state):
    if args.visual_prompt:
        if args.balanced_vp_prompt:
            prompt = PROMPT_VLT_VP_BALANCED
        else:
            prompt = PROMPT_VLT_VP_COMPACT if args.compact_prompt else PROMPT_VLT_VP
        vp_enable = True
    else:
        if args.balanced_prompt:
            prompt = PROMPT_VLT_BALANCED
        else:
            prompt = PROMPT_VLT_COMPACT if args.compact_prompt else PROMPT_VLT
        vp_enable = False
    search_bbox = state["current_bbox"]
    if vp_enable and args.vp_placement == "random":
        search_bbox = random_size_matched_bbox(
            search_bbox,
            state["img_width"],
            state["img_height"],
            args.vp_scale,
            args.vp_random_seed,
            state["template_image_path"],
            state["frame_index"],
        )
    return InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": prompt.replace(
                        "<language>", json.dumps(state["language"])
                    ).replace("<vp_color>", args.vp_color),
                }
            ],
            images=[state["template_image_path"], state["image_paths"][state["frame_index"]]],
            visual_prompt={
                "phase": "test",
                "enable": vp_enable,
                "scale": args.vp_scale,
                "color": args.vp_color,
                "width": args.vp_width,
                "line_style": args.vp_line_style,
                "opacity": args.vp_opacity,
                "template_bbox": state["template_bbox"],
                "template_scale": args.template_scale,
                "search_bbox": search_bbox,
                "save_image": args.save_image,
            },
        )


def parse_response(response, img_width, img_height, norm_coords=False):
    try:
        answer = response.split("</think>")[-1].strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(answer)
        if isinstance(parsed, dict):
            if str(parsed.get("visible", "yes")).lower() in {"no", "false", "0"}:
                return [0, 0, 0, 0], False
            bbox = parsed.get("bbox")
        else:
            bbox = parsed
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        assert len(bbox) == 4, f"bbox length is not 4: {bbox}"
        max_width, max_height = (1000, 1000) if norm_coords else (img_width, img_height)
        if not check_bbox(bbox, max_width, max_height):
            return [0, 0, 0, 0], False
        return bbox, True
    except Exception as e:
        print(f"Error response: {response}")
        print(f"Error: {e}")
        return [0, 0, 0, 0], False


def make_sequence_state(sample, save_path):
    video_name = sample["video_name"]
    image_paths = sample["image_paths"]
    language = sample["language"]
    bboxes = sample["bboxes"]
    init_bbox = bboxes[0]  # x,y,w,h
    template_bbox = [
        init_bbox[0],
        init_bbox[1],
        init_bbox[0] + init_bbox[2],
        init_bbox[1] + init_bbox[3],
    ]
    # x,y,w,h -> x1,y1,x2,y2
    current_bbox = [
        init_bbox[0],
        init_bbox[1],
        init_bbox[0] + init_bbox[2],
        init_bbox[1] + init_bbox[3],
    ]
    template_image_path = image_paths[0]
    img_width, img_height = Image.open(template_image_path).size
    return {
        "video_name": video_name,
        "image_paths": image_paths,
        "language": language,
        "template_bbox": template_bbox,
        "current_bbox": current_bbox,
        "result_bbox": [current_bbox],
        "template_image_path": template_image_path,
        "img_width": img_width,
        "img_height": img_height,
        "frame_index": 1,
        "save_path": save_path,
    }


def save_sequence(state, save_path):
    result_bbox = convert_x1y1x2y2_to_xywh(state["result_bbox"])
    result_path = os.path.join(save_path, f"{state['video_name']}.txt")
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        for bbox in result_bbox:
            f.write(f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n")  # x,y,w,h
    os.replace(tmp_path, result_path)


def build_dataset(args, finished_videos, dataset_name):
    # Independently scheduled shards may start at different times. Keep their
    # ownership stable by sharding the full sequence list before skipping work.
    constructor_finished_videos = [] if args.stable_sharding else finished_videos
    if dataset_name == "tnl2k":
        dataset = TNL2KDataset(
            root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnl2k/test",
            finished_videos=constructor_finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    elif dataset_name == "otb":
        dataset = OTBDataset(
            root_dir="/share/wangjingchao/track_datasets/OTB_sentences",
            finished_videos=constructor_finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    elif dataset_name == "tnllt":
        dataset = TNLLTDataset(
            root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnllt",
            finished_videos=constructor_finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    dataset.video_names = [name for name in dataset.video_names if name not in finished_videos]
    if args.sequence_list:
        with open(args.sequence_list, encoding="utf-8") as source:
            requested_videos = {line.strip() for line in source if line.strip()}
        dataset.video_names = [
            name for name in dataset.video_names if name in requested_videos
        ]
    return dataset


def round_robin_workloads(datasets):
    iterators = [(iter(dataset), save_path) for dataset, save_path in datasets]
    while iterators:
        remaining = []
        for iterator, save_path in iterators:
            try:
                yield next(iterator), save_path
                remaining.append((iterator, save_path))
            except StopIteration:
                pass
        iterators = remaining


def run_workloads(args, engine, request_config, datasets):
    pending = iter(round_robin_workloads(datasets))
    active = []
    total = sum(len(dataset) for dataset, _ in datasets)
    progress = tqdm(total=total, desc="Processing pooled datasets")

    while True:
        while len(active) < args.batch_size:
            try:
                sample, save_path = next(pending)
                active.append(make_sequence_state(sample, save_path))
            except StopIteration:
                break
        if not active:
            break

        requests = [build_infer_request(args, state) for state in active]
        responses = engine.infer(requests, request_config)
        next_active = []
        for state, response_obj in zip(active, responses):
            response = response_obj.choices[0].message.content
            bbox, success = parse_response(
                response, state["img_width"], state["img_height"], args.norm_coords
            )
            if success:
                state["current_bbox"] = (
                    denormalize_coordinates(bbox, state["img_width"], state["img_height"])
                    if args.norm_coords else bbox
                )
            state["result_bbox"].append(state["current_bbox"])
            state["frame_index"] += 1
            if state["frame_index"] >= len(state["image_paths"]):
                save_sequence(state, state["save_path"])
                progress.update(1)
            else:
                next_active.append(state)
        active = next_active
    progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="jcwang0602/VPTracker"
    )
    parser.add_argument(
        "--balanced_vp_prompt",
        action="store_true",
        help="Use the medium-length VP prompt matched to balanced VP training data.",
    )
    parser.add_argument(
        "--balanced_prompt",
        action="store_true",
        help="Use the medium-length non-VP prompt matched to balanced training data.",
    )
    parser.add_argument(
        "--model_name", type=str, default="VPTracker-Qwen3.5-2B"
    )
    parser.add_argument("--dataset", type=str, default="tnllt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--visual_prompt", action="store_true")
    parser.add_argument(
        "--compact_prompt",
        action="store_true",
        help="Use the compact tracking prompt matched to compact-prompt training data.",
    )
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--infer_backend", type=str, default="pt")
    parser.add_argument(
        "--model_type",
        choices=["qwen3_vl", "qwen3_5"],
        default="qwen3_5",
    )
    parser.add_argument("--seg_index", type=int, default=0)
    parser.add_argument("--seg_total", type=int, default=1)
    parser.add_argument("--stable_sharding", action="store_true")
    parser.add_argument(
        "--sequence_list",
        type=str,
        default=None,
        help="Optional newline-delimited sequence allowlist applied after completed results are skipped.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vp_scale", type=int, default=3)
    parser.add_argument("--vp_color", type=str, default="red")
    parser.add_argument("--vp_width", type=int, default=3)
    parser.add_argument("--vp_line_style", choices=["solid", "dashed"], default="solid")
    parser.add_argument("--vp_opacity", type=float, default=1.0)
    parser.add_argument(
        "--vp_placement", choices=["previous_prediction", "random"], default="previous_prediction"
    )
    parser.add_argument("--vp_random_seed", type=int, default=0)
    parser.add_argument("--template_scale", type=float, default=2.0)
    parser.add_argument("--tracking_results_dir", type=str, default=None)
    parser.add_argument("--norm_coords", type=bool, default=False)
    parser.add_argument("--save_vis_results", type=bool, default=False)
    parser.add_argument("--vis_results_dir", type=str, default="vis_results")
    args = parser.parse_args()

    request_config = RequestConfig(max_tokens=128, temperature=0)

    print(f"DATA INFO: save_dir: {args.save_dir}", flush=True)
    print(
        f"DATA INFO: seg_index: {args.seg_index}, seg_total: {args.seg_total}",
        flush=True,
    )
    _, processor = get_model_processor(
        args.checkpoint, load_model=False, model_type=args.model_type
    )
    template = get_template(processor, enable_thinking=False)
    engine = None
    engine = VllmEngine(
        args.checkpoint,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        torch_dtype=torch.bfloat16,
        model_type=args.model_type,
        template=template,
    )

    if args.dataset == "all":
        dataset_names = ["otb", "tnllt", "tnl2k"]
    elif args.dataset == "all_tracking":
        dataset_names = ["tnllt", "tnl2k"]
    else:
        dataset_names = [args.dataset]

    datasets = []
    for dataset_name in dataset_names:
        save_path = os.path.join(args.save_dir, args.model_name, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        # 查看已经跑完的视频
        if os.path.exists(args.save_dir) and not args.debug:
            finished_videos = [
                f.name.replace(".txt", "")
                for f in os.scandir(save_path)
                if f.is_file() and f.name.endswith(".txt")
            ]
            print(f'"DATA INFO: finished_videos: {len(finished_videos)}')
        else:
            os.makedirs(save_path, exist_ok=True)
            finished_videos = []
        dataset = build_dataset(args, finished_videos, dataset_name)
        datasets.append((dataset, save_path))
    run_workloads(args, engine, request_config, datasets)
