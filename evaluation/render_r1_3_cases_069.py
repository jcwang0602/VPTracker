#!/usr/bin/env python3
"""Render the R1.3 qualitative challenge figure with experiment 069 outputs."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnl2k/test")
ANNO_ROOT = ROOT / "TNL2K_evaluation_toolkit" / "annos" / "annos" / "gt_rect"
BASELINE_ROOT = ROOT / "result_tnl2k"
VP_RESULT_ROOT = ROOT / (
    "outputs_vpt_r1/results_batchpool/"
    "069_qwen35_2b_vp_blue_w1_op100_out25_pixelcoords_no_imgtok_cap_1m_lr2e-5_bs128_e1_pdbs4_ga4_pool/tnl2k"
)
OUTPUTS = [
    ROOT / "paper/assert/Attr_Case_01.png",
    ROOT / "paper/VPTracker_PRL/Attr_Case_01.png",
]

CASES = [
    (
        "Viewpoint Change",
        "Cartoon_ManHead_video_03_done",
        "Language: the head of the man who is walking on the street",
        [1474, 1475, 1476, 1477],
    ),
    (
        "Full Occlusion",
        "test_022_Monkey_video_01_done",
        "Language: the monkey",
        [419, 420, 421, 422],
    ),
    (
        "Out of View",
        "advSamp_CartoonHuLuWa_video_07_done",
        "Language: the animal in the cave",
        [870, 871, 872, 873],
    ),
]

COLORS = {
    "GT": (0, 0, 255),
    "VPTracker": (0, 200, 0),
    "CTVLT": (255, 110, 30),
    "JointNLT": (0, 165, 255),
    "TemTrack": (220, 40, 210),
    "UVLTrack": (220, 210, 40),
}
TRACKERS = ["GT", "VPTracker", "CTVLT", "JointNLT", "TemTrack", "UVLTrack"]
FONT_PATH = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"


def load_boxes(path):
    for delimiter in (",", "\t", None):
        try:
            return np.atleast_2d(np.loadtxt(path, delimiter=delimiter))[:, :4]
        except ValueError:
            continue
    raise ValueError(f"Unable to parse {path}")


def draw_box(image, box, color):
    x, y, width, height = np.rint(box).astype(int)
    if width > 0 and height > 0:
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), color, 3)


def make_panel(image_paths, frame_number, boxes, size):
    image_path = image_paths[frame_number - 1]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    frame_index = frame_number - 1
    for tracker in TRACKERS:
        draw_box(image, boxes[tracker][frame_index], COLORS[tracker])
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def text_width(draw, text, font):
    return draw.textbbox((0, 0), text, font=font)[2]


def main():
    panel_width, panel_height = 790, 525
    header_height, legend_height, margin, gutter = 78, 128, 12, 16
    canvas_width = margin * 2 + panel_width * 4 + gutter * 3
    canvas_height = header_height + (panel_height + header_height) * 3 + legend_height
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    for row, (challenge, sequence, language, frames) in enumerate(CASES):
        image_paths = sorted((DATA_ROOT / sequence / "imgs").iterdir())
        boxes = {"GT": load_boxes(ANNO_ROOT / f"{sequence}.txt")}
        boxes["VPTracker"] = load_boxes(VP_RESULT_ROOT / f"{sequence}.txt")
        for tracker in TRACKERS[2:]:
            boxes[tracker] = load_boxes(BASELINE_ROOT / tracker / f"{sequence}.txt")
        y_top = row * (panel_height + header_height) + header_height
        for column, frame_number in enumerate(frames):
            x = margin + column * (panel_width + gutter)
            canvas[y_top : y_top + panel_height, x : x + panel_width] = make_panel(
                image_paths, frame_number, boxes, (panel_width, panel_height)
            )

    # PIL gives the figure a publication-friendly Times-compatible typeface.
    figure = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(figure)
    header_font = ImageFont.truetype(FONT_PATH, 48)
    frame_font = ImageFont.truetype(FONT_PATH, 50)
    legend_font = ImageFont.truetype(FONT_PATH, 42)
    for row, (challenge, _, language, frames) in enumerate(CASES):
        y_top = row * (panel_height + header_height) + header_height
        draw.text((margin, y_top - 60), language, font=header_font, fill="black")
        draw.text(
            (canvas_width - margin - text_width(draw, challenge, header_font), y_top - 60),
            challenge,
            font=header_font,
            fill="black",
        )
        for column, frame_number in enumerate(frames):
            x = margin + column * (panel_width + gutter)
            draw.text((x + 18, y_top + 14), f"#{frame_number:04d}", font=frame_font, fill="red")

    legend_y = canvas_height - 74
    slot_width = canvas_width / len(TRACKERS)
    for index, tracker in enumerate(TRACKERS):
        label = tracker if tracker != "GT" else "Ground Truth"
        label_width = text_width(draw, label, legend_font)
        group_width = 74 + 18 + label_width
        x = int(slot_width * (index + 0.5) - group_width / 2)
        color = tuple(reversed(COLORS[tracker]))
        draw.rectangle((x, legend_y - 28, x + 74, legend_y + 8), fill=color)
        draw.text((x + 92, legend_y - 38), label, font=legend_font, fill="black")

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.save(output, compress_level=3)
        print(output)


if __name__ == "__main__":
    main()
