import glob
from typing import Any, Dict, List
import torch
import os
import pandas
import numpy as np
import cv2


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


class LaSOTDataset(BaseDataset):
    def __init__(
        self,
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/LaSOTBenchmark",
        split="train",
        need_language_new=True,
    ):
        super().__init__(root_dir)
        self.dataset_name = "lasot"
        self.dataset_type = "video"
        self.split = split
        if split == "train":
            self.data_split_path = "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data_specs/lasot_train_split.txt"
        else:
            self.data_split_path = "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data_specs/lasot_test_split.txt"
        self.need_language_new = need_language_new
        # 读取data_split_path中的视频名称
        with open(self.data_split_path, "r") as f:
            self.video_names = [line.strip() for line in f if line.strip()]

        self.root_dir = root_dir
        # 获取每个视频的图片路径
        self.get_image_paths()

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        self.video_names_to_image_paths = {}
        for video_name in self.video_names:
            video_dir = os.path.join(
                self.root_dir, video_name.split("-")[0], video_name
            )
            img_dir = os.path.join(video_dir, "img")
            image_names = sorted(os.listdir(img_dir))
            self.video_names_to_image_paths[video_name] = [
                os.path.join(img_dir, image_name) for image_name in image_names
            ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        language_path = os.path.join(
            self.root_dir, video_name.split("-")[0], video_name, "nlp.txt"
        )
        bboxes_path = os.path.join(
            self.root_dir,
            video_name.split("-")[0],
            video_name,
            "groundtruth.txt",
        )
        with open(language_path, "r") as f:
            language = f.read()
        if self.need_language_new:
            language_new_path = os.path.join(
                "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/descriptions_lasot_v2",
                f"{video_name}.txt",
            )
            with open(language_new_path, "r") as f:
                language_new = f.readlines()
        else:
            language_new = None
        with open(bboxes_path, "r") as f:
            bbox = [
                list(map(int, line.strip().split(",")))
                for line in f
                if line.strip()
            ]
        bbox = torch.tensor(bbox)  # x,y,w,h
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "language": language,
            "language_new": language_new,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


class GOT10KDataset(BaseDataset):
    def __init__(self, root_dir="/share/wangjingchao/track_datasets/got10k"):
        super().__init__(root_dir)
        self.dataset_type = "video"
        self.dataset_name = "got10k"
        self.train_splits = ["train", "val"]
        self.get_image_paths()

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        self.video_names = []
        self.video_names_to_image_paths = {}
        for train_split in self.train_splits:
            videos_dir = os.path.join(self.root_dir, train_split)
            for video_name in os.listdir(videos_dir):
                img_dir = os.path.join(videos_dir, video_name)
                # 如果是一个文件夹
                if os.path.isdir(img_dir):
                    self.video_names.append(video_name)
                    image_names = glob.glob(os.path.join(img_dir, "*.jpg"))
                    self.video_names_to_image_paths[video_name] = [
                        os.path.join(img_dir, image_name)
                        for image_name in image_names
                    ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        bboxes_path = os.path.join(
            os.path.dirname(images_path_list[0]), "groundtruth.txt"
        )
        bbox = pandas.read_csv(
            bboxes_path,
            delimiter=",",
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False,
        ).values
        bbox = torch.tensor(bbox)
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


class TrackingNetDataset(BaseDataset):
    def __init__(
        self, root_dir="/share/wangjingchao/track_datasets/TrackingNet"
    ):
        super().__init__(root_dir)
        self.dataset_type = "video"
        self.dataset_name = "trackingnet"
        self.train_splits = [
            "TRAIN_0",
            "TRAIN_1",
            "TRAIN_2",
            "TRAIN_3",
            "TRAIN_4",
            "TRAIN_5",
            "TRAIN_6",
            "TRAIN_7",
            "TRAIN_8",
            "TRAIN_9",
            "TRAIN_10",
            "TRAIN_11",
        ]

        # 获取每个视频的图片路径
        self.get_image_paths()

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        self.video_names = []
        self.video_names_to_image_paths = {}
        for train_split in self.train_splits:
            videos_dir = os.path.join(self.root_dir, train_split, "frames")
            for video_name in os.listdir(videos_dir):
                self.video_names.append(video_name)
                img_dir = os.path.join(videos_dir, video_name)
                image_names = sorted(
                    os.listdir(img_dir), key=lambda x: int(x.split(".")[0])
                )
                self.video_names_to_image_paths[video_name] = [
                    os.path.join(img_dir, image_name)
                    for image_name in image_names
                ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        bboxes_path = os.path.join(
            self.root_dir,
            images_path_list[0].split("/")[-4],
            "anno",
            f"{video_name}.txt",
        )
        bbox = pandas.read_csv(
            bboxes_path,
            delimiter=",",
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False,
        ).values
        bbox = torch.tensor(bbox)
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


class VastTrackDataset(BaseDataset):
    def __init__(
        self, root_dir="/share/wangjingchao/track_datasets/VastTrack/train"
    ):
        super().__init__(root_dir)
        self.dataset_name = "vasttrack"
        self.dataset_type = "video"

        self.root_dir = root_dir
        # 获取每个视频的图片路径
        self.get_image_paths()

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        self.video_names = []
        self.video_names_to_image_paths = {}
        self.class_names = os.listdir(self.root_dir)
        for class_name in self.class_names:
            video_names = os.listdir(os.path.join(self.root_dir, class_name))
            for video_name in video_names:
                self.video_names.append(video_name)
                video_dir = os.path.join(self.root_dir, class_name, video_name)
                img_dir = os.path.join(video_dir, "imgs")
                image_names = sorted(os.listdir(img_dir))
                self.video_names_to_image_paths[video_name] = [
                    os.path.join(img_dir, image_name)
                    for image_name in image_names
                ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        video_dir = os.path.dirname(os.path.dirname(images_path_list[0]))
        language_path = os.path.join(video_dir, "nlp.txt")
        bboxes_path = os.path.join(video_dir, "Groundtruth.txt")
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


class TNL2KDataset(BaseDataset):
    def __init__(
        self,
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnl2k",
        split="train",
        need_language_new=True,
    ):
        super().__init__(root_dir)
        self.dataset_name = "tnl2k"
        self.dataset_type = "video"
        self.split = split
        self.need_language_new = need_language_new
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
        language_new_path = os.path.join(
            "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/descriptions_v2",
            f"{video_name}.txt",
        )
        bboxes_path = os.path.join(self.root_dir, video_name, "groundtruth.txt")
        with open(language_path, "r") as f:
            language = f.read()
        if self.need_language_new:
            with open(language_new_path, "r") as f:
                language_new = f.readlines()
        else:
            language_new = None
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
            "language_new": language_new,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


class RefCOCODataset(BaseDataset):
    def __init__(
        self,
        root_dir="/share/wangjingchao/vg_data",
        split="train",
        version="2014",
        name="gref",
        splitBy="google",
    ):
        super().__init__(root_dir)
        self.dataset_type = "image"
        self.dataset_name = name
        self.split = split
        self.img_pth = os.path.join(root_dir, "{}{}".format("train", version))
        self.anno_path = os.path.join(
            root_dir, "{}/instances.json".format(name)
        )

        self.root_dir = root_dir
        # 获取每个视频的图片路径
        self.get_image_paths()

    def __len__(self):
        return len(self.img_names)

    def get_image_paths(self):
        self.im_dir = os.path.join(
            self.root_dir, "image_data", "mscoco", "images", "train2014"
        )
        self.imgset_info = []
        splits = [self.split]
        for split in splits:
            imgset_file = "{0}_{1}.pth".format(self.dataset_name, split)
            imgset_path = os.path.join(
                self.root_dir, "split_data", self.dataset_name, imgset_file
            )
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        # process the image set info
        self.img_names, _, self.bboxs, self.phrases, _ = zip(*self.imgset_info)
        self.bboxs = torch.tensor(self.bboxs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = os.path.join(self.im_dir, self.img_names[idx])
        return {
            "video_name": self.img_names[idx],
            "image_paths": img_path,
            "language": self.phrases[idx],
            "bboxes": self.bbox_xywh_to_x1y1x2y2(self.bboxs)[idx],
            "valid": True,
            "visible": True,
        }


class OTB99Dataset(BaseDataset):
    def __init__(
        self, root_dir="/share/wangjingchao/track_datasets/OTB_sentences"
    ):
        super().__init__(root_dir)
        self.dataset_name = "otb99"
        self.dataset_type = "video"
        self.root_dir = root_dir
        # 获取每个视频的图片路径
        self.get_image_paths()

    def __len__(self):
        return len(self.video_names)

    def get_image_paths(self):
        self.video_names = []
        self.video_names_to_image_paths = {}
        self.video_names = [
            txt_file.split(".")[0]
            for txt_file in os.listdir(
                os.path.join(self.root_dir, "OTB_query_train")
            )
        ]

        for video_name in self.video_names:
            video_dir = os.path.join(self.root_dir, "OTB_videos", video_name)
            img_dir = os.path.join(video_dir, "img")
            image_names = sorted(os.listdir(img_dir))
            self.video_names_to_image_paths[video_name] = [
                os.path.join(img_dir, image_name) for image_name in image_names
            ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_name = self.video_names[idx]
        images_path_list = self.video_names_to_image_paths[video_name]
        video_dir = os.path.dirname(os.path.dirname(images_path_list[0]))
        language_path = os.path.join(
            self.root_dir, "OTB_query_train", f"{video_name}.txt"
        )
        bboxes_path = os.path.join(video_dir, "groundtruth_rect.txt")
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
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnllt",
        split="train",
        need_language_new=True,
    ):
        super().__init__(root_dir)
        self.dataset_name = "tnllt"
        self.dataset_type = "video"
        self.need_language_new = need_language_new
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

        language_new_path = os.path.join(
            "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/descriptions_v2",
            f"{video_name}.txt",
        )
        bboxes_path = os.path.join(self.root_dir, video_name, "groundtruth.txt")
        with open(language_path, "r") as f:
            language = f.read()
        if self.need_language_new:
            with open(language_new_path, "r") as f:
                language_new = f.readlines()
        else:
            language_new = None
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
            "language_new": language_new,
            "bboxes": self.bbox_xywh_to_x1y1x2y2(bbox),
            "valid": self.get_valid_by_bbox(bbox),
            "visible": self.get_visible_by_bbox(bbox),
        }


if __name__ == "__main__":
    dataset = TNL2KDataset()
    # dataset = LaSOTDataset()
    # dataset = TrackingNetDataset()
    # dataset = GOT10KDataset()
    # dataset = VastTrackDataset()
    # dataset = OTB99Dataset()
    # dataset = RefCOCODataset()
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
