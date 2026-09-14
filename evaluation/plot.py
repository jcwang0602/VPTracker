import os
import numpy as np
import sys

sys.path.append("/mnt/shared-storage-user/mineru4s/jcwang/VPLT")
from utils.config_plot_style import config_plot_style
from utils.config_sequence import config_sequence
from utils.eval_tracker import eval_tracker
from utils.plot_draw_save import plot_draw_save

# Define paths  配置路径
tmp_mat_path = "tmp_mat/"
path_anno = "annos"
path_att = "annos/att/"
seq_eval_config = "sequence_evaluation_config/"
rp_all = "results/"  # 结果保存路径
save_fig_path = "evaluation/result_fig/"
save_fig_suf = "png"

# Attribute names and figure names 每个挑战的名称和生成图片的名称
att_name = [
    "ALL",
    "Camera Motion",
    "Rotation",
    "Deformation",
    "Full Occlusion",
    "Illumination Variation",
    "Out-of-View",
    "Partial Occlusion",
    "Viewpoint Change",
    "Scale Variation",
    "Background Clutter",
    "Motion Blur",
    "Aspect Ration Change",
    "Low Resolution",
    "Fast Motion",
    "Adversarial Samples",
    "Thermal Crossover",
    "Modality Switch",
]
att_fig_name = ["ALL", "CM", "ROT", "DEF", "FOC", "IV", "OV", "POC", "VC", "SV", "BC", "MB", "ARC", "LR", "FM", "AS", "TC", "MS"]

# Evaluation dataset type 评估测试集的类型
evaluation_dataset_type = "TNL2K_testing_set"

# use normalization or not 是否使用归一化
norm_dst = True

# 获取跟踪器列表
result_model_name = [
    "OpenGVLab_InternVL2_5-1B",
]

trackers = []
for model_name in result_model_name:
    trackers.append({"name": model_name, "publish": "OpenGVLab"})

# 获取视频序列列表
sequences = config_sequence(evaluation_dataset_type, seq_eval_config)
# 获取绘制风格
plot_style = config_plot_style()
# 获取视频序列的长度和跟踪器的数量
num_seq = len(sequences)
num_tracker = len(trackers)
# 获取所有跟踪器的名称
name_tracker_all = [trackers[i]["name"] for i in range(num_tracker)]
# 获取所有序列的挑战矩阵
name_seq_all = []
att_all = np.zeros((num_seq, len(att_name)))
for i in range(num_seq):
    name_seq_all.append(sequences[i])
    seq_att = np.loadtxt(os.path.join(path_att, f"{sequences[i]}_attribute.txt"))
    att_all[i, :] = seq_att

# 设置用于评估的参数
metric_type_set = ["overlap"]  # ['error', 'overlap']
eval_type = "OPE"
ranking_type = "AUC"
rank_num = 50

# Threshold sets for error and overlap 设置阈值
threshold_set_error = np.arange(0, 51) / 100 if norm_dst else np.arange(0, 51)
threshold_set_overlap = np.arange(0, 1.05, 0.05)

for metric_type in metric_type_set:
    if metric_type == "error":
        threshold_set = threshold_set_error
        rank_idx = 21
        x_label_name = "Location error threshold"
        y_label_name = "Precision"
    elif metric_type == "overlap":
        threshold_set = threshold_set_overlap
        rank_idx = 11
        x_label_name = "Overlap threshold"
        y_label_name = "Success rate"

    # Skipped evaluation logic for brevity...
    if metric_type == "overlap" and ranking_type == "threshold":
        continue

    t_num = len(threshold_set)
    plot_type = metric_type + "_" + eval_type

    if metric_type == "error":
        title_name = "Precision plots of " + eval_type
        if norm_dst:
            title_name = "Normalized " + title_name

        if evaluation_dataset_type == "all":
            title_name += " on TNL2K"
        else:
            title_name += " on TNL2K Testing Set"
    elif metric_type == "overlap":
        title_name = "Success plots of " + eval_type
        if evaluation_dataset_type == "all":
            title_name += " on TNL2K"
        else:
            title_name += " on TNL2K Testing Set"

    dataName = os.path.join(tmp_mat_path, f"aveSuccessRatePlot_{num_tracker}alg_{plot_type}.npz")

    # Evaluate tracker performance
    if not os.path.exists(dataName):
        eval_tracker(sequences, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst)

    # Load data
    loaded_data = np.load(dataName)
    ave_success_rate_plot = loaded_data["ave_success_rate_plot"]  # Load the required variable

    num_tracker = ave_success_rate_plot.shape[0]

    # Check and set rank_num
    if rank_num > num_tracker or rank_num < 0:
        rank_num = num_tracker

    fig_name = f"{plot_type}_{ranking_type}"
    idx_seq_set = np.arange(len(sequences))

    plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)

    att_trld = 0
    att_num = att_all.shape[1]

    for att_idx in range(att_num):
        idx_seq_set = np.where(att_all[:, att_idx] > att_trld)[0]
        if len(idx_seq_set) < 2:
            continue

        print(f"{att_name[att_idx]} {len(idx_seq_set)}")

        fig_name = f"{att_fig_name[att_idx]}_{plot_type}_{ranking_type}"
        title_name = f"Plots of {eval_type}: {att_name[att_idx]} ({len(idx_seq_set)})"

        if metric_type == "overlap":
            title_name = f"Success plots of {eval_type} - {att_name[att_idx]} ({len(idx_seq_set)})"
        elif metric_type == "error":
            title_name = f"Precision plots of {eval_type} - {att_name[att_idx]} ({len(idx_seq_set)})"
            if norm_dst:
                title_name = f"Normalized {title_name}"

        plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
