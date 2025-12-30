import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from evaluation.extract_results import extract_results
from evaluation.tnl2k_dataset import TNL2KDataset
from evaluation.tnllt_dataset import TNLLTDataset


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[
        valid_sequence, :, :
    ]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[
        valid_sequence, :, :
    ]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score


def generate_formatted_report(row_labels, scores, table_name=""):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = "\n{label: <{width}} |".format(
        label=table_name, width=name_width
    )

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = "{prev} {s: <{width}} |".format(
            prev=report_text, s=s, width=s_w
        )

    report_text = "{prev}\n".format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = "{prev}{tracker: <{width}} |".format(
            prev=report_text, tracker=d_name, width=name_width
        )
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = "{prev} {score: <{width}} |".format(
                prev=report_text,
                score="{:0.2f}".format(score_value[trk_id].item()),
                width=s_w,
            )
        report_text = "{prev}\n".format(prev=report_text)

    return report_text


def get_tracker_display_name(tracker):
    if tracker["disp_name"] is None:
        if tracker["run_id"] is None:
            disp_name = "{}_{}".format(tracker["name"], tracker["param"])
        else:
            disp_name = "{}_{}_{:03d}".format(
                tracker["name"], tracker["param"], tracker["run_id"]
            )
    else:
        disp_name = tracker["disp_name"]

    return disp_name


def print_results(
    tracking_results_dir,
    trackers,
    dataset,
    plot_types=("success", "prec", "norm_prec"),
    skip_missing_seq=False,
    **kwargs,
):
    """Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = extract_results(
        tracking_results_dir,
        trackers,
        dataset,
        **kwargs,
        skip_missing_seq=skip_missing_seq,
    )

    tracker_names = eval_data["trackers"]
    valid_sequence = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)

    print(
        "\nReporting results over {} / {} sequences".format(
            valid_sequence.long().sum().item(), valid_sequence.shape[0]
        )
    )

    scores = {}

    # ********************************  Success Plot **************************************
    if "success" in plot_types:
        threshold_set_overlap = torch.tensor(eval_data["threshold_set_overlap"])
        ave_success_rate_plot_overlap = torch.tensor(
            eval_data["ave_success_rate_plot_overlap"]
        )

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(
            ave_success_rate_plot_overlap, valid_sequence
        )
        scores["AUC"] = auc
        scores["SUC"] = auc_curve[:, threshold_set_overlap == 0.50]
        # scores["OP75"] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if "prec" in plot_types:
        ave_success_rate_plot_center = torch.tensor(
            eval_data["ave_success_rate_plot_center"]
        )

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(
            ave_success_rate_plot_center, valid_sequence
        )
        scores["Precision"] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if "norm_prec" in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(
            eval_data["ave_success_rate_plot_center_norm"]
        )

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(
            ave_success_rate_plot_center_norm, valid_sequence
        )
        scores["Norm Precision"] = norm_prec_score

    # Print
    report_text = generate_formatted_report(
        tracker_names, scores, table_name="TNLLT"
    )
    print(report_text)


if __name__ == "__main__":
    dataset_name = "tnllt"  # tnl2k,tnllt
    tracking_results_dir = "outputs/results"
    tracker_names = [
        "vptracker"
    ]
    if dataset_name == "tnl2k":
        dataset = TNL2KDataset()
    elif dataset_name == "tnllt":
        dataset = TNLLTDataset()
    print_results(
        tracking_results_dir, tracker_names, dataset, skip_missing_seq=False
    )
