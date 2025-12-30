import os
from PIL import Image, ImageDraw


def save_tracking_results(args, img_path, bbox, video_name):
    save_path = os.path.join(args.tracking_results_dir, video_name)
    os.makedirs(save_path, exist_ok=True)
    # 读取图片
    img = Image.open(img_path)
    # 绘制矩形框
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline="red", width=2)
    # 保存图片
    img.save(os.path.join(save_path, f"{img_path.split('/')[-1].split('.')[0]}.jpg"))


def draw_bbox(image: Image, bbox: list[int], color: str = "red", width: int = 2) -> Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=width)
    return image
