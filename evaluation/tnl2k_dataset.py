from typing import Any, Dict, List
import torch
import os


class TNL2KDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnl2k/test",
        finished_videos=[],
        seg_index=None,
        seg_total=1,
        debug=False,
    ):
        self.dataset_name = "tnl2k"
        self.root_dir = root_dir
        # 获取所有的视频名称
        self.video_names = os.listdir(root_dir)
        # 过滤掉已经跑完的视频
        self.video_names = [
            video_name
            for video_name in self.video_names
            if video_name not in finished_videos
        ]
        # 获取每个视频的图片路径
        self.video_names_to_image_paths = self._get_image_paths(
            self.video_names
        )
        # 根据每个视频的长度对self.video_names进行排序，从最长到最短
        self.video_names = sorted(
            self.video_names,
            key=lambda x: len(self.video_names_to_image_paths[x]),
            reverse=not debug,
        )
        # 按照seg_index和seg_total对self.video_names进行分组
        self.video_names = self.video_names[seg_index::seg_total]

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
        target_visible = None
        return {
            "video_name": video_name,
            "image_paths": images_path_list,
            "language": language,
            "bboxes": bbox,
            "target_visible": target_visible,
        }


if __name__ == "__main__":
    dataset = TNL2KDataset(
        root_dir="/mnt/shared-storage-user/mineru4s/jcwang/VPLT/data/tnl2k",
        finished_videos=["CartoonXiYouJi_video_03"],
    )
    for i in range(len(dataset)):
        print(f"{dataset[i]['video_name']}, {len(dataset[i]['image_paths'])}")
    print(len(dataset))
