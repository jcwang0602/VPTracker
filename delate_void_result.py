import os
from pathlib import Path


def delete_empty_txt_files(results_path):
    """
    删除results文件夹及其子文件夹下所有空的txt文件
    """
    results_path = Path(results_path)

    if not results_path.exists():
        print("results文件夹不存在")
        return

    print("开始删除空的txt文件:")
    print("=" * 60)

    deleted_count = 0

    # 遍历results文件夹及其子文件夹下所有txt文件
    for txt_file in results_path.rglob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            # 内容为空则删除
            if not content:
                txt_file.unlink()
                print(f"已删除空内容文件: {txt_file}")
                deleted_count += 1
        except Exception as e:
            print(f"无法删除文件: {txt_file}, 错误: {e}")

    print("=" * 60)
    print(f"共删除空的txt文件: {deleted_count}")


if __name__ == "__main__":
    results_path = "/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/outputs/results"
    delete_empty_txt_files(results_path)
