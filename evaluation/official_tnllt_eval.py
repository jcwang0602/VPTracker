"""Python port of the official TNLLT MATLAB OPE evaluation toolkit.

Source: Event-AHU/Open_VLTrack, ``TNLLT_Evaluation_Toolkit/utils``.  TNLLT
uses the opposite absent-label convention from TNL2K: ``absent == 0`` means
that the target is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.official_tnl2k_eval import (
    NORMALIZED_PRECISION_THRESHOLDS,
    OVERLAP_THRESHOLDS,
    RAW_PRECISION_THRESHOLDS,
    load_matrix,
)


def _load_visibility(sequence_dir: Path, ground_truth: np.ndarray) -> np.ndarray:
    """Load official labels when available, otherwise reconstruct from zero GT boxes."""
    label_path = sequence_dir / "absent_label.txt"
    if label_path.is_file():
        visibility = np.asarray(np.loadtxt(label_path), dtype=float).reshape(-1)
        if len(visibility) != len(ground_truth):
            raise ValueError(f"Visibility/GT length mismatch: {sequence_dir.name}")
        return visibility
    # The local TNLLT copy stores absent frames as 0,0,0,0.  This reconstructs
    # the official absent_anno whose visible values are 1 and absent values 0.
    return ((ground_truth[:, 2] > 0) & (ground_truth[:, 3] > 0)).astype(float)


def _repair_predictions(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    sequence_length = len(ground_truth)
    if len(prediction) < sequence_length:
        raise ValueError(
            f"Prediction has {len(prediction)} frames but GT has {sequence_length}; "
            "the official MATLAB evaluator cannot evaluate a short result."
        )
    prediction = prediction[:sequence_length, :4].copy()
    for frame in range(1, sequence_length):
        row = prediction[frame]
        if (
            not np.isfinite(row).all()
            or row[2] <= 0
            or row[3] <= 0
        ) and not np.isnan(ground_truth[frame]).any():
            prediction[frame] = prediction[frame - 1]
    prediction[0] = ground_truth[0]
    return prediction


def sequence_curves(
    prediction: np.ndarray, ground_truth: np.ndarray, visibility: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal NumPy counterpart of TNLLT ``calc_seq_err_robust.m``/``eval_tracker.m``."""
    ground_truth = ground_truth[:, :4].copy()
    if len(visibility) != len(ground_truth):
        raise ValueError("TNLLT visibility-label and GT lengths differ")
    sequence_length = len(ground_truth)
    prediction = _repair_predictions(prediction, ground_truth)

    keep = visibility != 0  # Official TNLLT convention.
    prediction = prediction[keep]
    ground_truth = ground_truth[keep]

    pred_center = prediction[:, :2] + (prediction[:, 2:] - 1.0) / 2.0
    gt_center = ground_truth[:, :2] + (ground_truth[:, 2:] - 1.0) / 2.0
    raw_error = np.linalg.norm(pred_center - gt_center, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_error = np.linalg.norm(
            pred_center / ground_truth[:, 2:] - gt_center / ground_truth[:, 2:], axis=1
        )

    valid_gt = (ground_truth > 0).all(axis=1)
    raw_error[~valid_gt] = -1.0
    normalized_error[~valid_gt] = -1.0
    left_top = np.maximum(prediction[:, :2], ground_truth[:, :2])
    right_bottom = np.minimum(
        prediction[:, :2] + prediction[:, 2:] - 1.0,
        ground_truth[:, :2] + ground_truth[:, 2:] - 1.0,
    )
    intersection = np.maximum(right_bottom - left_top + 1.0, 0.0).prod(axis=1)
    union = prediction[:, 2:].prod(axis=1) + ground_truth[:, 2:].prod(axis=1) - intersection
    overlap = np.full(len(ground_truth), -1.0)
    overlap[valid_gt] = np.divide(
        intersection[valid_gt], union[valid_gt], out=np.zeros(valid_gt.sum()), where=union[valid_gt] != 0
    )

    return (
        np.array([(overlap > threshold).sum() / sequence_length for threshold in OVERLAP_THRESHOLDS]),
        np.array([(raw_error <= threshold).sum() / sequence_length for threshold in RAW_PRECISION_THRESHOLDS]),
        np.array(
            [(normalized_error <= threshold).sum() / sequence_length for threshold in NORMALIZED_PRECISION_THRESHOLDS]
        ),
    )


def evaluate_trackers(
    results_root: Path,
    trackers: list[str],
    data_root: Path = Path("/mnt/shared-storage-user/mineru4s/jcwang/VPTrack/data/tnllt"),
    official_root: Path = Path("third_party/TNLLT_official/TNLLT_Evaluation_Toolkit"),
) -> dict[str, dict[str, float]]:
    """Evaluate complete tracker outputs on the official TNLLT 50-sequence test split."""
    sequences = [line.strip() for line in (official_root / "sequence_evaluation_config" / "testing_set.txt").read_text().splitlines() if line.strip()]
    metrics: dict[str, dict[str, float]] = {}
    for tracker in trackers:
        success, raw_precision, normalized_precision = [], [], []
        for name in sequences:
            sequence_dir = data_root / name
            ground_truth = load_matrix(sequence_dir / "groundtruth.txt")
            visibility = _load_visibility(sequence_dir, ground_truth)
            prediction = load_matrix(results_root / tracker / "tnllt" / f"{name}.txt")
            curves = sequence_curves(prediction, ground_truth, visibility)
            success.append(curves[0])
            raw_precision.append(curves[1])
            normalized_precision.append(curves[2])
        success_curve = np.mean(np.stack(success), axis=0) * 100.0
        raw_curve = np.mean(np.stack(raw_precision), axis=0) * 100.0
        normalized_curve = np.mean(np.stack(normalized_precision), axis=0) * 100.0
        metrics[tracker] = {
            "AUC": float(success_curve.mean()),
            "SUC": float(success_curve[10]),
            "PR": float(raw_curve[20]),
            "NPR": float(normalized_curve[20]),
        }
    return metrics
