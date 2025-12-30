from typing import Any, Dict, List
import torch
import os
import numpy as np



class BaseDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset_type = "video"
        self.dataset_name = "base"
        self.video_names = []

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        pass

    def bbox_xywh_to_x1y1x2y2(self, bbox: torch.Tensor) -> list[list[int]]:
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        # x1,y1,小于 0 的设置为 0
        x1 = torch.where(x1 < 0, torch.zeros_like(x1), x1)
        y1 = torch.where(y1 < 0, torch.zeros_like(y1), y1)
        x2 = bbox[:, 0] + bbox[:, 2]
        y2 = bbox[:, 1] + bbox[:, 3]
        # 转为int
        return torch.stack([x1, y1, x2, y2], dim=1).int().numpy().tolist()

    def get_valid_by_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        return ((bbox[:, 2] > 0) & (bbox[:, 3] > 0)).numpy().tolist()

    def get_visible_by_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        return ((bbox[:, 2] > 0) & (bbox[:, 3] > 0)).byte().numpy().tolist()

    def prase_new_language(self, language: str) -> str:
        result = []
        for line in language.splitlines():
            line = line.strip()
            if not line:
                continue
            num_str, text = line.split(" ", 1)
            result.extend([text] * 100)
        return result



class TNL2KDataset(BaseDataset):
    def __init__(
        self,
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnl2k",
        split="train",
    ):
        super().__init__(root_dir)
        self.dataset_name = "tnl2k"
        self.dataset_type = "video"
        self.split = split
        if split == "train":
            self.root_dir = os.path.join(root_dir, "train")
            print(f"train root_dir: {self.root_dir}")
        else:
            self.root_dir = os.path.join(root_dir, "test")
            print(f"test root_dir: {self.root_dir}")
        # 获取所有的视频名称
        self.video_names = os.listdir(self.root_dir)
        # 获取每个视频的图片路径
        self.video_names_to_image_paths = self._get_image_paths(
            self.video_names
        )

    def __len__(self):
        return len(self.video_names)

    def _get_image_paths(self, video_names: List[str]) -> Dict[str, List[str]]:
        video_names_to_image_paths = {}
        for video_name in video_names:
            video_dir = os.path.join(self.root_dir, video_name)
            img_dir = os.path.join(video_dir, "imgs")
            image_names = sorted(os.listdir(img_dir))
            video_names_to_image_paths[video_name] = [
                os.path.join(img_dir, image_name) for image_name in image_names
            ]
        return video_names_to_image_paths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        language_path = os.path.join(self.root_dir, video_name, "language.txt")
        bboxes_path = os.path.join(self.root_dir, video_name, "groundtruth.txt")
        with open(language_path, "r") as f:
            language = f.read()
        with open(bboxes_path, "r") as f:
            bbox = [
                list(map(int, line.strip().split(",")))
                for line in f
                if line.strip()
            ]
        bbox = torch.tensor(bbox)
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "language": language,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


class TNLLTDataset(BaseDataset):
    def __init__(
        self,
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnllt",
        split="train",
    ):
        super().__init__(root_dir)
        self.dataset_name = "tnllt"
        self.dataset_type = "video"
        self.root_dir = root_dir
        # 获取所有的视频名称
        if split == "train":
            split_path = "data_specs/tnllt_train_split.txt"
        else:
            split_path = "data_specs/tnllt_test_split.txt"
        with open(split_path, "r") as f:
            self.video_names = [line.strip() for line in f if line.strip()]
        # 获取每个视频的图片路径
        self.video_names_to_image_paths = self._get_image_paths(
            self.video_names
        )

    def __len__(self):
        return len(self.video_names)

    def _get_image_paths(self, video_names: List[str]) -> Dict[str, List[str]]:
        video_names_to_image_paths = {}
        for video_name in video_names:
            video_dir = os.path.join(self.root_dir, video_name)
            img_dir = os.path.join(video_dir, "imgs")
            image_names = sorted(os.listdir(img_dir))
            video_names_to_image_paths[video_name] = [
                os.path.join(img_dir, image_name) for image_name in image_names
            ]
        return video_names_to_image_paths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        language_path = os.path.join(self.root_dir, video_name, "language.txt")
        bboxes_path = os.path.join(self.root_dir, video_name, "groundtruth.txt")
        with open(language_path, "r") as f:
            language = f.read()
        with open(bboxes_path, "r") as f:
            bbox = [
                list(map(float, line.strip().split(",")))
                for line in f
                if line.strip()
            ]
            # 将 bbox使用 numpy转成 int
            bbox = np.array(bbox, dtype=np.int32)
        bbox = torch.tensor(bbox)
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "language": language,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


if __name__ == "__main__":
    dataset = TNL2KDataset()
    # dataset = TNLLTDataset()
    error_num = 0
    for i in range(len(dataset)):
        try:
            print(
                f"{dataset[i]['video_name']}, {len(dataset[i]['image_paths'])}, {dataset[i]['language']}"
            )
            pass
        except Exception as e:
            error_num += 1
            print(e)
    print(len(dataset), error_num)
