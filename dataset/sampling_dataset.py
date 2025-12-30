import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bbox_utils import normalize_coordinates

import random
import jsonlines
from dataset.track_dataset import (
    RefCOCODataset,
    OTB99Dataset,
    TNL2KDataset,
    LaSOTDataset,
    GOT10KDataset,
    TNLLTDataset,
    TrackingNetDataset,
    VastTrackDataset,
)
from tqdm import tqdm


def sample_pair_from_video_qwen_vlt(
    template_image_path,
    search_image_path,
    language,
    normalize_search_bbox,
    template_bbox,
    search_bbox,
    template_scale,
    visual_prompt,
    vp_color,
    vp_width,
    vp_inbbox_ratio,
    visible,
):
    if visual_prompt:
        # TODO: 要有一定概率，给的 VP 要比真实的物体小
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a visual object tracker.
Given the visual information of the target object <template><image></template>, and the language description of the target object <ref> {language.strip()} </ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max]. 

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.

Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.
""",
                },
                {
                    "role": "assistant",
                    "content": f'```json\n{{\n  "visible": "{visible}",\n "bbox": {normalize_search_bbox}\n}}\n```',
                },
            ],
            "images": [template_image_path, search_image_path],
            "visual_prompt": {
                "phase": "train",
                "enable": True,
                "scale": vp_scale,
                "color": vp_color,
                "width": vp_width,
                "inbbox_ratio": vp_inbbox_ratio,
                "template_bbox": template_bbox,
                "template_scale": template_scale,
                "search_bbox": search_bbox,
            },
        }
    else:
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a visual object tracker.
Given the visual information of the target object <template><image></template>, and the language description of the target object <ref> {language.strip()} </ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max]. 

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.
""",
                },
                {
                    "role": "assistant",
                    "content": f'```json\n{{\n  "visible": "{visible}",\n  "bbox": {normalize_search_bbox}\n}}\n```',
                },
            ],
            "images": [template_image_path, search_image_path],
            "visual_prompt": {
                "phase": "train",
                "enable": False,
                "template_bbox": template_bbox,
                "search_bbox": search_bbox,
                "template_scale": template_scale,
            },
        }
    return jsonl_data


def sample_visible_ids(
    visible,
    num_ids=1,
    min_id=None,
    max_id=None,
    allow_invisible=False,
    force_invisible=False,
):
    if num_ids == 0:
        return []
    if min_id is None or min_id < 0:
        min_id = 0
    if max_id is None or max_id > len(visible):
        max_id = len(visible)
    # get valid ids
    if force_invisible:
        valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
    else:
        if allow_invisible:
            valid_ids = [i for i in range(min_id, max_id)]
        else:
            valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
        return None

    return random.choices(valid_ids, k=num_ids)


def sample_seq_from_dataset(dataset):
    # 获取数据集长度
    dataset_length = len(dataset)
    # 随机选择一个视频
    video_idx = random.randint(0, dataset_length - 1)
    # 随机选择一个视频序列
    seq = dataset[video_idx]
    return seq


def sample_jsonl_from_dataset(
    dataset,
    num_template_frames=1,
    num_search_frames=1,
    template_scale=2.0,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    norm_coords=False,
):
    for _ in range(10):
        dataset = random.choices(datasets, dataset_ratios, k=1)[0]
        if dataset.dataset_type == "video":  # 如果是视频数据集
            # 随机选择一个视频
            seq = None
            while not seq:
                try:
                    seq = sample_seq_from_dataset(dataset)
                except:
                    continue
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0
            # 采样基础模板
            base_frame_id = sample_visible_ids(
                seq["visible"],
                num_ids=1,
                min_id=num_template_frames - 1,
                max_id=len(seq["visible"]) - num_search_frames,
            )
            # 采样历史模板
            prev_frame_ids = sample_visible_ids(
                seq["visible"],
                num_ids=num_template_frames - 1,
                min_id=0,
                max_id=base_frame_id[0],
            )
            if prev_frame_ids is None:
                gap_increase += 5
                continue
            # 模板帧
            template_frame_ids = base_frame_id + prev_frame_ids
            # 搜索帧
            search_frame_ids = sample_visible_ids(
                seq["visible"],
                min_id=template_frame_ids[0] + 1,
                max_id=len(seq["visible"]),
                num_ids=num_search_frames,
                allow_invisible=True,
            )
            # Increase gap until a frame is found
            gap_increase += 5
        else:  # 如果是图像数据集
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * num_template_frames
            search_frame_ids = [1] * num_search_frames

        # 获取模板帧和搜索帧的地址
        template_path = seq["image_paths"][template_frame_ids[0]]
        search_path = seq["image_paths"][search_frame_ids[0]]
        template_bbox = seq["bboxes"][template_frame_ids[0]]
        search_bbox = seq["bboxes"][search_frame_ids[0]]
        language = seq["language"]
        visible = "yes" if seq["visible"][search_frame_ids[0]] else "no"
        if norm_coords:
            normalize_search_bbox = normalize_coordinates(search_bbox, seq["width"], seq["height"])
        else:
            normalize_search_bbox = search_bbox
        # 从 seq中抽取一个模板图像和一个搜索图像
        jsonl = sample_pair_from_video_qwen_vlt(
            template_image_path=template_path,
            search_image_path=search_path,
            language=language,
            normalize_search_bbox=normalize_search_bbox,
            template_bbox=template_bbox,
            search_bbox=search_bbox,
            template_scale=template_scale,
            visual_prompt=visual_prompt,
            vp_color=vp_color,
            vp_width=vp_width,
            vp_inbbox_ratio=vp_inbbox_ratio,
            visible=visible,
        )
        if jsonl is not None:
            return jsonl
    return None


def process_dataset_vt_qwen(
    datasets,
    dataset_ratios,
    save_path,
    template_scale,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    sample_num=1,
    norm_coords=False,
):
    num_template_frames = 1
    num_search_frames = 1
    jsonls_list = []
    for i in tqdm(range(sample_num)):
        try:
            jsonl = sample_jsonl_from_dataset(
                random.choices(datasets, dataset_ratios, k=1)[0],
                num_template_frames,
                num_search_frames,
                template_scale,
                visual_prompt,
                vp_color,
                vp_width,
                vp_inbbox_ratio,
                norm_coords,
            )
        except:
            jsonl = None
        if jsonl is not None:
            jsonls_list.append(jsonl)
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls_list)
    print(f"Success count: {sample_num}, complated!!!")


if __name__ == "__main__":
    # ATCTrack 3M样本
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tnllt")
    parser.add_argument("--dataset_ratio", type=str, default="1")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--template_scale", type=float, default=2.0)
    parser.add_argument("--visual_prompt", type=bool, default=False)
    parser.add_argument("--vp_inbbox_ratio", type=float, default=0.95)
    parser.add_argument("--vp_color", type=str, default="red")
    parser.add_argument("--vp_width", type=int, default=3)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--norm_coords", type=bool, default=False)
    parser.add_argument(
        "--save_path",
        type=str,
        default="/share/wangjingchao/VPLT/data/01_qwen_vlt_10w.jsonl",
    )
    args = parser.parse_args()

    import time

    dataset_list = [x.strip() for x in args.dataset.split(",")]
    dataset_ratios = [int(x.strip()) for x in args.dataset_ratio.split(",")]
    datasets = []
    for dataset_name in dataset_list:
        start_time = time.time()
        if dataset_name == "tnl2k":
            datasets.append(TNL2KDataset(root_dir='path_to_tnl2k',split=args.phase))
            elapsed = time.time() - start_time
            print(
                f"Loaded {len(datasets)} datasets: {dataset_name} with {len(datasets[-1])} sequences, load time: {elapsed:.2f}s",
                flush=True,
            )
        elif dataset_name == "tnllt":
            datasets.append(TNLLTDataset(root_dir='path_to_tnllt',split=args.phase))
            elapsed = time.time() - start_time
            print(
                f"Loaded {len(datasets)} datasets: {dataset_name} with {len(datasets[-1])} sequences, load time: {elapsed:.2f}s",
                flush=True,
            )
    # 打印参数
    print("=" * 100)
    print(f"visual_prompt: {args.visual_prompt}")
    print(f"vp_color: {args.vp_color}")
    print(f"vp_width: {args.vp_width}")
    print(f"vp_inbbox_ratio: {args.vp_inbbox_ratio}")
    print(f"sample_num: {args.sample_num}")
    print(f"norm_coords: {args.norm_coords}")
    print(f"save_path: {args.save_path}")
    print(f"template_scale: {args.template_scale}")
    print(f"phase: {args.phase}")
    print(f"dataset: {args.dataset}")
    print(f"dataset_ratio: {args.dataset_ratio}")
    print("=" * 100)
    start_time = time.time()
    process_dataset_vt_qwen(
        datasets,
        dataset_ratios,
        save_path=args.save_path,
        template_scale=args.template_scale,
        visual_prompt=args.visual_prompt,
        vp_color=args.vp_color,
        vp_width=args.vp_width,
        vp_inbbox_ratio=args.vp_inbbox_ratio,
        sample_num=args.sample_num,
        norm_coords=args.norm_coords,
    )
    elapsed = time.time() - start_time
    print(f"Sampling time: {elapsed:.2f}s")
