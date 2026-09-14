"""Portable readers for the TNL2K and TNLLT training splits."""

from array import array
from dataclasses import dataclass
from pathlib import Path
import os
import re


TNLLT_TRAIN_SPLIT = Path(__file__).resolve().parents[1] / "data_specs/tnllt_train_split.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sequence:
    directory: Path
    frames: list[str]
    boxes: array
    visible: bytearray
    template_indices: array
    language: str

    def image_path(self, index: int) -> str:
        return str(self.directory / "imgs" / self.frames[index])

    def bbox(self, index: int) -> list[int]:
        return list(self.boxes[index * 4:index * 4 + 4])


def read_sequence(directory: Path) -> Sequence:
    """Read xywh annotations using the original integer conversion and visibility."""
    directory = directory.resolve()
    with os.scandir(directory / "imgs") as entries:
        frames = sorted(entry.name for entry in entries
                        if Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS and entry.is_file())
    boxes = array("i")
    visible = bytearray()
    with (directory / "groundtruth.txt").open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            values = re.split(r"[,\s]+", line.strip())
            if len(values) != 4:
                raise ValueError(f"{directory}/groundtruth.txt:{line_number}: expected x,y,w,h")
            # TNLLT originally cast xywh to integers before converting to xyxy.
            x, y, width, height = (int(float(value)) for value in values)
            boxes.extend((max(0, x), max(0, y), x + width, y + height))
            visible.append(width > 0 and height > 0)
    if len(frames) != len(visible):
        raise ValueError(f"{directory}: {len(frames)} images but {len(visible)} boxes")
    template_indices = array("i", (index for index in range(len(frames) - 1) if visible[index]))
    language = (directory / "language.txt").read_text(encoding="utf-8-sig").strip()
    if not language:
        raise ValueError(f"{directory}/language.txt: empty target description")
    return Sequence(directory, frames, boxes, visible, template_indices, language)


def read_split(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{path}: training split must be nonempty and have no duplicates")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError(f"{path}: expected sequence names, not paths")
    return names


def load_training_sequences(tnl2k_root: Path, tnllt_root: Path,
                            tnllt_split: Path = TNLLT_TRAIN_SPLIT) -> tuple[list[Sequence], list[Sequence]]:
    """TNL2K always reads train/; TNLLT accepts only the supplied training list or a subset."""
    train_root = tnl2k_root.resolve() / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"TNL2K training directory is missing: {train_root}")
    allowed = set(read_split(TNLLT_TRAIN_SPLIT))
    tnllt_names = read_split(tnllt_split)
    unknown = set(tnllt_names) - allowed
    if unknown:
        raise ValueError(f"TNLLT split contains sequences outside the published training split: {sorted(unknown)}")
    directories = (
        sorted(path for path in train_root.iterdir() if path.is_dir()),
        [tnllt_root / name for name in tnllt_names],
    )
    datasets = []
    for name, paths in zip(("TNL2K", "TNLLT"), directories):
        sequences = []
        for path in paths:
            sequence = read_sequence(path)
            if sequence.template_indices:
                sequences.append(sequence)
            else:
                print(f"Skipping {path}: no visible template with a later search frame", flush=True)
        if not sequences:
            raise ValueError(f"{name}: no usable training sequences")
        print(f"Loaded {name}: {len(sequences)} training sequences", flush=True)
        datasets.append(sequences)
    return datasets[0], datasets[1]
