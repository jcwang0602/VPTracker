import json
import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
from swift.llm import (
    PtEngine,
    VllmEngine,
    RequestConfig,
    InferRequest,
)
from evaluation.tnl2k_dataset import TNL2KDataset
from evaluation.otb_dataset import OTBDataset
from evaluation.tnllt_dataset import TNLLTDataset
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bbox_utils import convert_x1y1x2y2_to_xywh, denormalize_coordinates
from utils.vis_utils import save_tracking_results
import re


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


def run_model(
    args,
    engine,
    request_config,
    template_image_path,
    search_image_path,
    language,
    init_bbox,
    current_bbox,
    video_name,
    img_width,
    img_height,
):
    if args.visual_prompt:
        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a visual object tracker.
Given the visual information of the target object, including an initial visual template <template><image></template>, 
and the language description of the target object, including an initial language description <ref>{language}</ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max]. 

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.

Please note that you should first search for the target within the {args.vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.""",
                    }
                ],
                images=[template_image_path, search_image_path],
                visual_prompt={
                    "phase": "test",
                    "enable": True,
                    "scale": args.vp_scale,
                    "color": args.vp_color,
                    "width": args.vp_width,
                    "template_bbox": init_bbox,
                    "template_scale": args.template_scale,
                    "search_bbox": current_bbox,
                    "save_image": args.save_image,
                },
            )
        ]
    else:
        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a visual object tracker.
Given the visual information of the target object, including an initial visual template <template><image></template>, 
and the language description of the target object, including an initial language description <ref>{language}</ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max]. 

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.
""",
                    }
                ],
                images=[template_image_path, search_image_path],
                visual_prompt={
                    "phase": "train",
                    "enable": False,
                    "template_bbox": init_bbox,
                    "template_scale": args.template_scale,
                    "search_bbox": current_bbox,
                },
            )
        ]
    resp_list = engine.infer(infer_requests, request_config)
    response = resp_list[0].choices[0].message.content
    success = True
    # print("*" * 100)
    # print(f"Response: {response}")
    try:
        answer = (
            re.search(r"```json\s*(.*?)\s*```", response, re.S).group(1).strip()
        )
        answer = json.loads(answer)
        visible = answer["visible"]
        bbox = answer["bbox"]
        if visible == "no":
            bbox = [0, 0, 0, 0]
            success = False
        else:
            if isinstance(bbox, str):
                bbox = json.loads(bbox)
            success = True
        assert len(bbox) == 4, f"bbox length is not 4: {bbox}"
        # bbox 中的值都大于 0 并且 x2 y2 大于 x1 y1
        if not check_bbox(bbox, img_width, img_height):
            bbox = [0, 0, 0, 0]
            success = False
    except Exception as e:
        print(f"Error response: {response}")
        print(f"Error: {e}")
        bbox = [0, 0, 0, 0]
        success = False
    if args.tracking_results_dir:
        save_tracking_results(args, search_image_path, bbox, video_name)
    return bbox, success


def run_sequence(
    args,
    engine,
    request_config,
    video_name,
    image_paths,
    language,
    bboxes,
    save_path,
):
    # 创建保存结果的列表
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
    result_bbox = [current_bbox]
    template_image_path = image_paths[0]
    img_width, img_height = Image.open(template_image_path).size
    # 遍历所有的图像
    for i, search_image_path in enumerate(image_paths[1:]):
        bbox, success = run_model(
            args,
            engine,
            request_config,
            template_image_path,
            search_image_path,
            language,
            template_bbox,
            current_bbox,
            video_name,
            img_width,
            img_height,
        )
        if args.debug:
            print(f"Video: {search_image_path}, Response: {bbox}")
        if success:
            if args.norm_coords:
                current_bbox = denormalize_coordinates(
                    bbox, img_width, img_height
                )
            else:
                current_bbox = bbox
        result_bbox.append(current_bbox)
    result_bbox = convert_x1y1x2y2_to_xywh(result_bbox)
    with open(os.path.join(save_path, f"{video_name}.txt"), "w") as f:
        for bbox in result_bbox:
            f.write(f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n")  # x,y,w,h


def collate_fn(batches):
    video_name = batches[0]["video_name"]
    images_path_list = batches[0]["image_paths"]
    language = batches[0]["language"]
    bboxes = batches[0]["bboxes"]
    return video_name, images_path_list, language, bboxes


def run_datasets(
    args, engine, request_config, finished_videos, dataset_name, save_path
):
    if dataset_name == "tnl2k":
        dataset = TNL2KDataset(
            root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnl2k/test",
            finished_videos=finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    elif dataset_name == "otb":
        dataset = OTBDataset(
            root_dir="/share/wangjingchao/track_datasets/OTB_sentences",
            finished_videos=finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    elif dataset_name == "tnllt":
        dataset = TNLLTDataset(
            root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnllt",
            finished_videos=finished_videos,
            seg_index=args.seg_index,
            seg_total=args.seg_total,
            debug=args.debug,
        )

    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for video_name, image_paths, language, bboxes in tqdm(
        dataloader, desc="Processing datasets"
    ):
        run_sequence(
            args,
            engine,
            request_config,
            video_name,
            image_paths,
            language,
            bboxes,
            save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="OpenGVLab/InternVL2_5-1B"
    )
    parser.add_argument(
        "--model_name", type=str, default="OpenGVLab/InternVL2_5-1B"
    )
    parser.add_argument("--dataset", type=str, default="tnllt")
    parser.add_argument("--batch-size", type=int, default=1)
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
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--infer_backend", type=str, default="pt")
    parser.add_argument("--seg_index", type=int, default=0)
    parser.add_argument("--seg_total", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vp_scale", type=int, default=3)
    parser.add_argument("--vp_color", type=str, default="red")
    parser.add_argument("--vp_width", type=int, default=3)
    parser.add_argument("--template_scale", type=float, default=2.0)
    parser.add_argument("--tracking_results_dir", type=str, default=None)
    parser.add_argument("--norm_coords", type=bool, default=False)
    parser.add_argument("--save_vis_results", type=bool, default=False)
    parser.add_argument("--vis_results_dir", type=str, default="vis_results")
    args = parser.parse_args()

    request_config = RequestConfig(max_tokens=1024, temperature=0)

    print(f"DATA INFO: save_dir: {args.save_dir}", flush=True)
    print(
        f"DATA INFO: seg_index: {args.seg_index}, seg_total: {args.seg_total}",
        flush=True,
    )

    engine = None
    if args.infer_backend == "pt":
        engine = PtEngine(
            args.checkpoint, max_batch_size=1, attn_impl="flash_attn"
        )
    elif args.infer_backend == "vllm":
        engine = VllmEngine(
            args.checkpoint,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            torch_dtype=torch.bfloat16,
            model_type="qwen3_vl",
        )
    elif args.infer_backend == "sglang":
        engine = SglangEngine(
            model_id_or_path=args.checkpoint, torch_dtype=torch.bfloat16
        )
    elif args.infer_backend == "lmdeploy":
        engine = LmdeployEngine(args.checkpoint)

    if args.dataset == "all":
        dataset_names = ["otb", "tnllt", "tnl2k"]
    else:
        dataset_names = [args.dataset]

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
        run_datasets(
            args,
            engine,
            request_config,
            finished_videos,
            dataset_name,
            save_path,
        )
