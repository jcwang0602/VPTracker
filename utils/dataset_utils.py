import os
import random
import jsonlines
from tqdm import tqdm
from PIL import Image
from utils.bbox_utils import convert_xywh_to_x1y1x2y2, normalize_coordinates


def merge_video_to_datasets(video_path, save_path):
    jsonl_files = [f for f in os.listdir(video_path) if f.endswith(".jsonl")]
    data_lines = []
    # 遍历所有jsonl文件，统计总数量
    for jsonl_file in tqdm(jsonl_files):
        with jsonlines.open(os.path.join(video_path, jsonl_file)) as reader:
            data_lines.extend(reader)
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(data_lines)
    print(f"Total count: {len(data_lines)}")


# 获取排序后的图像路径列表
def get_sorted_image_paths(video_path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    try:
        all_files = os.listdir(video_path)
        image_files = [f for f in all_files if os.path.splitext(f.lower())[1] in valid_extensions]
        image_files_sorted = sorted(image_files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
        absolute_paths = [os.path.abspath(os.path.join(video_path, f)) for f in image_files_sorted]
        return absolute_paths
    except Exception as e:
        print(f"Error encountered during image path processing: {str(e)}")
        return []


# 读取文本文件中的边界框坐标
def read_txt_to_2d_list(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # 去掉空行
        lines = [line for line in lines if line.strip()]
        return [[float(num) for num in line.strip().split(",")] for line in lines]


def sample_pair_from_video_internvl(image_paths, bboxes, language, visual_prompt=False, img_width=None, img_height=None, skip_empty=True):
    # bboxes 是[x1,y1,x2,y2]格式
    images_num = len(image_paths)
    # 从图片中一次性随机2个图片，一个座位模板，一个座位搜索区域
    template_index, search_index = random.sample(range(images_num), 2)
    if bboxes[template_index] == [0.0, 0.0, 0.0, 0.0]:
        return None, False
    if skip_empty and bboxes[search_index] == [0.0, 0.0, 0.0, 0.0]:
        return None, False
    template_bbox = bboxes[template_index]
    search_bbox = bboxes[search_index]
    template_image_path = image_paths[template_index]
    search_image_path = image_paths[search_index]

    search_bbox_normalized = normalize_coordinates(search_bbox, img_width, img_height)
    template_bbox_normalized = normalize_coordinates(template_bbox, img_width, img_height)
    if visual_prompt:
        vp_color = "red"
        vp_width = 3
        vp_inbbox_ratio = 1.0
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please identify the target object specified by the bounding box <box>{template_bbox_normalized}</box> and the sentence describes: <ref>{language}</ref> in template image <image>, then locate it in search image <image> and return the bounding box coordinate in [x_min,y_min,x_max,y_max] format. Please note that the {vp_color} rectangular box should be searched first, and if not found, then other places should be searched.",
                },
                {"role": "assistant", "content": f"<ref>{language}</ref><box>[{search_bbox_normalized}]</box>"},
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
                "search_bbox": search_bbox,
            },
        }
    else:
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please identify the target specified by the bounding box <box>{template_bbox_normalized}</box> and the sentence describes: <ref>{language}</ref> in template image <image>, then locate it in search image <image> and return the bounding box coordinate in [x_min,y_min,x_max,y_max] format.",
                },
                {"role": "assistant", "content": f"<ref>{language}</ref><box>[{search_bbox_normalized}]</box>"},
            ],
            "images": [template_image_path, search_image_path],
            "visual_prompt": {
                "phase": "train",
                "enable": False,
                "template_bbox": template_bbox,
                "search_bbox": search_bbox,
            },
        }
    return jsonl_data, True


def sample_pair_from_video_qwen(
    image_paths,
    bboxes,
    language,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    img_width=None,
    img_height=None,
    skip_empty=True,
):
    # bboxes 是[x1,y1,x2,y2]格式
    images_num = len(image_paths)
    # 从图片中一次性随机2个图片，一个座位模板，一个座位搜索区域
    template_index, search_index = random.sample(range(images_num), 2)
    if bboxes[template_index] == [0, 0, 0, 0]:
        return None, False
    if skip_empty and bboxes[search_index] == [0, 0, 0, 0]:
        return None, False
    template_bbox = bboxes[template_index]
    search_bbox = bboxes[search_index]
    # 将bbox转成整数
    template_image_path = image_paths[template_index]
    search_image_path = image_paths[search_index]

    if visual_prompt:
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please identify the target object specified by the bounding box {template_bbox} and the sentence describes: {language} in <image>, then locate it in <image> and return [x_min,y_min,x_max,y_max] coordinates of the target object. Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
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
                "search_bbox": search_bbox,
            },
        }
    else:
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please identify the target object specified by the bounding box {template_bbox} and the sentence describes: {language} in <image>, then locate it in <image> and return [x_min,y_min,x_max,y_max] coordinates of the target object.",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
            ],
            "images": [template_image_path, search_image_path],
            "visual_prompt": {
                "phase": "train",
                "enable": False,
                "template_bbox": template_bbox,
                "search_bbox": search_bbox,
            },
        }
    return jsonl_data, True


def sample_pair_from_video_qwen_v2(
    image_paths,
    bboxes,
    language,
    template_scale=1.0,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    img_width=None,
    img_height=None,
    skip_empty=True,
):
    # bboxes 是[x1,y1,x2,y2]格式
    images_num = len(image_paths)
    # 从图片中一次性随机2个图片，一个座位模板，一个座位搜索区域
    template_index, search_index = random.sample(range(images_num), 2)
    if bboxes[template_index] == [0, 0, 0, 0]:
        return None, False
    if skip_empty and bboxes[search_index] == [0, 0, 0, 0]:
        return None, False
    template_bbox = bboxes[template_index]
    search_bbox = bboxes[search_index]
    # 将bbox转成整数
    template_image_path = image_paths[template_index]
    search_image_path = image_paths[search_index]

    if visual_prompt:
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a vision-language tracker.
Given:
a template image that shows the target object: <image>,
a natural language description: {language},

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].
Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.
""",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
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
                    "content": f"""You are a vision-language tracker.
Given:
a template image that shows the target object: <image>,
a natural language description: {language},

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].""",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
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
    return jsonl_data, True


def sample_pair_from_video_internvl_v2(
    image_paths,
    bboxes,
    language,
    template_scale=1.0,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    img_width=None,
    img_height=None,
    skip_empty=True,
):
    # bboxes 是[x1,y1,x2,y2]格式
    images_num = len(image_paths)
    # 从图片中一次性随机2个图片，一个座位模板，一个座位搜索区域
    template_index, search_index = random.sample(range(images_num), 2)
    if bboxes[template_index] == [0, 0, 0, 0]:
        return None, False
    if skip_empty and bboxes[search_index] == [0, 0, 0, 0]:
        return None, False
    template_bbox = bboxes[template_index]
    search_bbox = bboxes[search_index]
    template_bbox_normalized = normalize_coordinates(template_bbox, img_width, img_height)
    search_bbox_normalized = normalize_coordinates(search_bbox, img_width, img_height)
    # 将bbox转成整数
    template_image_path = image_paths[template_index]
    search_image_path = image_paths[search_index]

    if visual_prompt:
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a vision-language tracker.
Given:
a template image that shows the target object: <image>,
a natural language description: {language},

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].
Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.
""",
                },
                {"role": "assistant", "content": f"{search_bbox_normalized}"},
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
                    "content": f"""You are a vision-language tracker.
Given:
a template image that shows the target object: <image>,
a natural language description: {language},

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].""",
                },
                {"role": "assistant", "content": f"{search_bbox_normalized}"},
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
    return jsonl_data, True


def sample_pair_from_video_qwen_vt(
    image_paths,
    bboxes,
    language,
    template_scale=1.0,
    visual_prompt=False,
    vp_color="red",
    vp_width=3,
    vp_inbbox_ratio=1.0,
    img_width=None,
    img_height=None,
    skip_empty=True,
):
    # bboxes 是[x1,y1,x2,y2]格式
    images_num = len(image_paths)
    # 从图片中一次性随机2个图片，一个座位模板，一个座位搜索区域
    template_index, search_index = random.sample(range(images_num), 2)
    if bboxes[template_index] == [0, 0, 0, 0]:
        return None, False
    if skip_empty and bboxes[search_index] == [0, 0, 0, 0]:
        return None, False
    template_bbox = bboxes[template_index]
    search_bbox = bboxes[search_index]
    # 将bbox转成整数
    template_image_path = image_paths[template_index]
    search_image_path = image_paths[search_index]

    if visual_prompt:
        vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
        jsonl_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a visual tracker.
Given:
a template image that shows the target object: <image>,

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].
Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.
""",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
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
                    "content": f"""You are a visual tracker.
Given:
a template image that shows the target object: <image>,

locate the target object in <image>.
Return the bounding box in the format [x_min, y_min, x_max, y_max].""",
                },
                {"role": "assistant", "content": f"{search_bbox}"},
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
    return jsonl_data, True


def convert_to_jsonl_internvl(image_paths, bboxes, language, save_path, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, img_width=None, img_height=None, skip_empty=True):
    jsonls = []
    for image_path, bbox in zip(image_paths, bboxes):
        if skip_empty and bbox == [0.0, 0.0, 0.0, 0.0]:
            continue
        bbox_normalized = normalize_coordinates(bbox, img_width, img_height)
        if visual_prompt:
            vp_color = vp_color
            vp_width = vp_width
            vp_inbbox_ratio = vp_inbbox_ratio
            vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
            jsonl = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please identify the target specified by the sentence describes: <ref>{language}</ref>, in search image <image> and return the bounding box coordinate in [x_min,y_min,x_max,y_max] format. Please note that the {vp_color} rectangular box should be searched first, and if not found, then other places should be searched.",
                    },
                    {"role": "assistant", "content": f"<ref>{language}</ref><box>[{bbox_normalized}]</box>"},
                ],
                "images": [image_path],
                "visual_prompt": {
                    "phase": "train",
                    "enable": True,
                    "scale": vp_scale,
                    "color": vp_color,
                    "width": vp_width,
                    "inbbox_ratio": vp_inbbox_ratio,
                    "search_bbox": bbox,
                },
            }
        else:
            jsonl = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please identify the target specified by the sentence describes: <ref>{language}</ref> in <image> and return the bounding box coordinate in [x_min,y_min,x_max,y_max] format.",
                    },
                    {"role": "assistant", "content": f"<ref>{language}</ref><box>[{bbox_normalized}]</box>"},
                ],
                "images": [image_path],
                "visual_prompt": {
                    "phase": "train",
                    "enable": False,
                },
            }
        jsonls.append(jsonl)
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)


def convert_to_jsonl_qwen(image_paths, bboxes, language, save_path, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, img_width=None, img_height=None, skip_empty=True):
    jsonls = []
    for image_path, bbox in zip(image_paths, bboxes):
        if skip_empty and bbox == [0.0, 0.0, 0.0, 0.0]:
            continue
        if visual_prompt:
            vp_color = vp_color
            vp_width = vp_width
            vp_inbbox_ratio = vp_inbbox_ratio
            vp_scale = random.randint(2, 8)  # 从 2 - 8 随机选择一个
            jsonl = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please identify the target object specified by the sentence describes: {language}, then locate it in <image> and return [x_min,y_min,x_max,y_max] coordinates of the target. Please note that you should first search for the target within the {vp_color} rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box.",
                    },
                    {"role": "assistant", "content": f"{bbox}"},
                ],
                "images": [image_path],
                "visual_prompt": {
                    "phase": "train",
                    "enable": True,
                    "scale": vp_scale,
                    "color": vp_color,
                    "width": vp_width,
                    "inbbox_ratio": vp_inbbox_ratio,
                    "search_bbox": bbox,
                },
            }
        else:
            jsonl = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please identify the target object specified by the sentence describes: {language}, then locate it in <image> and return [x_min,y_min,x_max,y_max] coordinates of the target.",
                    },
                    {"role": "assistant", "content": f"{bbox}"},
                ],
                "images": [image_path],
                "visual_prompt": {
                    "phase": "train",
                    "enable": False,
                },
            }
        jsonls.append(jsonl)
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)


def process_dataset(dataset_path, video_name_list, base_save_path, model_name, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, skip_empty=False):
    # 处理每个视频序列
    success_count = 0
    total_count = len(video_name_list)
    for video_name in tqdm(video_name_list):
        try:
            img_dir = os.path.join(dataset_path, video_name, "imgs")
            image_paths = [os.path.abspath(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]
            # 读取图像获得高和宽
            img = Image.open(image_paths[0])
            img_width, img_height = img.size
            # 获取 bbox 和 language
            bbox_path = os.path.join(dataset_path, video_name, "groundtruth.txt")
            # 读取 bbox swift 需要[x1,y1,x2,y2]格式
            bboxes = read_txt_to_2d_list(bbox_path)
            if len(image_paths) != len(bboxes):
                continue
            bboxes = convert_xywh_to_x1y1x2y2(bboxes)
            # 将bbox转成整数
            bboxes = [list(map(int, bbox)) for bbox in bboxes]
            # 读取 language
            language_path = os.path.join(dataset_path, video_name, "language.txt")
            with open(language_path, "r") as f:
                language = f.read()
            save_path = os.path.join(base_save_path, f"{video_name}.jsonl")
            # 处理图像序列生成训练样本
            if model_name == "internvl":
                convert_to_jsonl_internvl(
                    image_paths,
                    bboxes,
                    language,
                    save_path,
                    img_width=img_width,
                    img_height=img_height,
                    visual_prompt=visual_prompt,
                    vp_color=vp_color,
                    vp_width=vp_width,
                    vp_inbbox_ratio=vp_inbbox_ratio,
                    skip_empty=skip_empty,
                )
            elif model_name == "qwen":
                convert_to_jsonl_qwen(
                    image_paths,
                    bboxes,
                    language,
                    save_path,
                    img_width=img_width,
                    img_height=img_height,
                    visual_prompt=visual_prompt,
                    vp_color=vp_color,
                    vp_width=vp_width,
                    vp_inbbox_ratio=vp_inbbox_ratio,
                    skip_empty=skip_empty,
                )
            success_count += 1
        except Exception as e:
            print(f"Error encountered during video processing: {str(video_name)}")
            continue
    print(f"Success count: {success_count}, Total count: {total_count}")


def process_dataset_vlt(dataset_path, video_name_list, save_path, model_name, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, skip_empty=False, sample_num=1):
    # 处理每个视频序列
    success_count = 0
    total_count = len(video_name_list)
    dataset_info = {}
    print(f"读取数据集信息，共{total_count}个视频")
    for video_name in video_name_list:
        img_dir = os.path.join(dataset_path, video_name, "imgs")
        image_paths = [os.path.abspath(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]

        img = Image.open(image_paths[0])
        img_width, img_height = img.size
        # 获取 bbox 和 language
        bbox_path = os.path.join(dataset_path, video_name, "groundtruth.txt")
        # 读取 bbox swift 需要[x1,y1,x2,y2]格式
        bboxes = read_txt_to_2d_list(bbox_path)
        if len(image_paths) != len(bboxes):
            continue
        bboxes = convert_xywh_to_x1y1x2y2(bboxes)
        # 将bbox转成整数
        bboxes = [list(map(int, bbox)) for bbox in bboxes]
        # 读取 language
        language_path = os.path.join(dataset_path, video_name, "language.txt")
        with open(language_path, "r") as f:
            language = f.read()

        dataset_info[video_name] = {
            "image_paths": image_paths,
            "img_width": img_width,
            "img_height": img_height,
            "bboxes": bboxes,
            "language": language,
        }
    print(f"读取数据集信息完成，共{total_count}个视频")
    print(f"开始采样，共{sample_num}个样本")
    jsonls = []
    while success_count < sample_num:
        # 随机选择一个视频
        video_name = random.choice(list(dataset_info.keys()))
        if model_name == "internvl":
            jsonl, success = sample_pair_from_video_internvl(
                image_paths=dataset_info[video_name]["image_paths"],
                bboxes=dataset_info[video_name]["bboxes"],
                language=dataset_info[video_name]["language"],
                visual_prompt=visual_prompt,
                img_width=dataset_info[video_name]["img_width"],
                img_height=dataset_info[video_name]["img_height"],
                skip_empty=skip_empty,
            )
        elif model_name == "qwen":
            jsonl, success = sample_pair_from_video_qwen(
                image_paths=dataset_info[video_name]["image_paths"],
                bboxes=dataset_info[video_name]["bboxes"],
                language=dataset_info[video_name]["language"],
                visual_prompt=visual_prompt,
                vp_color=vp_color,
                vp_width=vp_width,
                vp_inbbox_ratio=vp_inbbox_ratio,
                img_width=dataset_info[video_name]["img_width"],
                img_height=dataset_info[video_name]["img_height"],
                skip_empty=skip_empty,
            )
        if success:
            jsonls.append(jsonl)
            success_count += 1
            print(f"采样成功，当前成功数: {success_count}")
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)
    print(f"Success count: {success_count}, Total count: {total_count}")


def process_dataset_vlt_qwen(dataset_path, video_name_list, save_path, template_scale, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, skip_empty=False, sample_num=1):
    # 处理每个视频序列
    success_count = 0
    total_count = len(video_name_list)
    dataset_info = {}
    print(f"读取数据集信息，共{total_count}个视频")
    for video_name in video_name_list:
        img_dir = os.path.join(dataset_path, video_name, "imgs")
        image_paths = [os.path.abspath(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]

        img = Image.open(image_paths[0])
        img_width, img_height = img.size
        # 获取 bbox 和 language
        bbox_path = os.path.join(dataset_path, video_name, "groundtruth.txt")
        # 读取 bbox swift 需要[x1,y1,x2,y2]格式
        bboxes = read_txt_to_2d_list(bbox_path)
        if len(image_paths) != len(bboxes):
            continue
        bboxes = convert_xywh_to_x1y1x2y2(bboxes)
        # 将bbox转成整数
        bboxes = [list(map(int, bbox)) for bbox in bboxes]
        # 读取 language
        language_path = os.path.join(dataset_path, video_name, "language.txt")
        with open(language_path, "r") as f:
            language = f.read()

        dataset_info[video_name] = {
            "image_paths": image_paths,
            "img_width": img_width,
            "img_height": img_height,
            "bboxes": bboxes,
            "language": language,
        }
    print(f"读取数据集信息完成，共{total_count}个视频")
    print(f"开始采样，共{sample_num}个样本")
    jsonls = []
    while success_count < sample_num:
        # 随机选择一个视频
        video_name = random.choice(list(dataset_info.keys()))
        jsonl, success = sample_pair_from_video_qwen_v2(
            image_paths=dataset_info[video_name]["image_paths"],
            bboxes=dataset_info[video_name]["bboxes"],
            language=dataset_info[video_name]["language"],
            template_scale=template_scale,
            visual_prompt=visual_prompt,
            vp_color=vp_color,
            vp_width=vp_width,
            vp_inbbox_ratio=vp_inbbox_ratio,
            img_width=dataset_info[video_name]["img_width"],
            img_height=dataset_info[video_name]["img_height"],
            skip_empty=skip_empty,
        )
        if success:
            jsonls.append(jsonl)
            success_count += 1
            print(f"采样成功，当前成功数: {success_count}")
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)
    print(f"Success count: {success_count}, Total count: {total_count}")


def process_dataset_vt_qwen(dataset_path, video_name_list, save_path, template_scale, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, skip_empty=False, sample_num=1):
    # 处理每个视频序列
    success_count = 0
    total_count = len(video_name_list)
    dataset_info = {}
    print(f"读取数据集信息，共{total_count}个视频")
    for video_name in video_name_list:
        img_dir = os.path.join(dataset_path, video_name, "imgs")
        image_paths = [os.path.abspath(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]

        img = Image.open(image_paths[0])
        img_width, img_height = img.size
        # 获取 bbox 和 language
        bbox_path = os.path.join(dataset_path, video_name, "groundtruth.txt")
        # 读取 bbox swift 需要[x1,y1,x2,y2]格式
        bboxes = read_txt_to_2d_list(bbox_path)
        if len(image_paths) != len(bboxes):
            continue
        bboxes = convert_xywh_to_x1y1x2y2(bboxes)
        # 将bbox转成整数
        bboxes = [list(map(int, bbox)) for bbox in bboxes]
        # 读取 language
        language_path = os.path.join(dataset_path, video_name, "language.txt")
        with open(language_path, "r") as f:
            language = f.read()

        dataset_info[video_name] = {
            "image_paths": image_paths,
            "img_width": img_width,
            "img_height": img_height,
            "bboxes": bboxes,
            "language": language,
        }
    print(f"读取数据集信息完成，共{total_count}个视频")
    print(f"开始采样，共{sample_num}个样本")
    jsonls = []
    while success_count < sample_num:
        # 随机选择一个视频
        video_name = random.choice(list(dataset_info.keys()))
        jsonl, success = sample_pair_from_video_qwen_vt(
            image_paths=dataset_info[video_name]["image_paths"],
            bboxes=dataset_info[video_name]["bboxes"],
            language=dataset_info[video_name]["language"],
            template_scale=template_scale,
            visual_prompt=visual_prompt,
            vp_color=vp_color,
            vp_width=vp_width,
            vp_inbbox_ratio=vp_inbbox_ratio,
            img_width=dataset_info[video_name]["img_width"],
            img_height=dataset_info[video_name]["img_height"],
            skip_empty=skip_empty,
        )
        if success:
            jsonls.append(jsonl)
            success_count += 1
            print(f"采样成功，当前成功数: {success_count}")
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)
    print(f"Success count: {success_count}, Total count: {total_count}")


def process_dataset_vlt_internvl(dataset_path, video_name_list, save_path, template_scale, visual_prompt=False, vp_color="red", vp_width=3, vp_inbbox_ratio=1.0, skip_empty=False, sample_num=1):
    # 处理每个视频序列
    success_count = 0
    total_count = len(video_name_list)
    dataset_info = {}
    print(f"读取数据集信息，共{total_count}个视频")
    for video_name in video_name_list:
        img_dir = os.path.join(dataset_path, video_name, "imgs")
        image_paths = [os.path.abspath(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]

        img = Image.open(image_paths[0])
        img_width, img_height = img.size
        # 获取 bbox 和 language
        bbox_path = os.path.join(dataset_path, video_name, "groundtruth.txt")
        # 读取 bbox swift 需要[x1,y1,x2,y2]格式
        bboxes = read_txt_to_2d_list(bbox_path)
        if len(image_paths) != len(bboxes):
            continue
        bboxes = convert_xywh_to_x1y1x2y2(bboxes)
        # 将bbox转成整数
        bboxes = [list(map(int, bbox)) for bbox in bboxes]
        # 读取 language
        language_path = os.path.join(dataset_path, video_name, "language.txt")
        with open(language_path, "r") as f:
            language = f.read()

        dataset_info[video_name] = {
            "image_paths": image_paths,
            "img_width": img_width,
            "img_height": img_height,
            "bboxes": bboxes,
            "language": language,
        }
    print(f"读取数据集信息完成，共{total_count}个视频")
    print(f"开始采样，共{sample_num}个样本")
    jsonls = []
    while success_count < sample_num:
        # 随机选择一个视频
        video_name = random.choice(list(dataset_info.keys()))
        jsonl, success = sample_pair_from_video_internvl_v2(
            image_paths=dataset_info[video_name]["image_paths"],
            bboxes=dataset_info[video_name]["bboxes"],
            language=dataset_info[video_name]["language"],
            template_scale=template_scale,
            visual_prompt=visual_prompt,
            vp_color=vp_color,
            vp_width=vp_width,
            vp_inbbox_ratio=vp_inbbox_ratio,
            img_width=dataset_info[video_name]["img_width"],
            img_height=dataset_info[video_name]["img_height"],
            skip_empty=skip_empty,
        )
        if success:
            jsonls.append(jsonl)
            success_count += 1
            print(f"采样成功，当前成功数: {success_count}")
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(jsonls)
    print(f"Success count: {success_count}, Total count: {total_count}")
