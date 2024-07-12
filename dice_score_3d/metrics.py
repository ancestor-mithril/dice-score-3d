import json
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence, Tuple

import numpy as np
from numpy import ndarray

from readers import read_mask


def dice_metrics(ground_truths: str, predictions: str, output_path: str, reorient: bool, dtype: str, prefix: str,
                 suffix: str, indices: dict, num_workers: int, console: bool):
    dtype = np.uint8 if dtype == 'uint8' else np.uint16
    assert os.path.isfile(ground_truths) and os.path.isfile(predictions) or \
           os.path.isdir(ground_truths) and os.path.isdir(predictions), ('Prediction path and GT path must both be a '
                                                                         'a single file or a folder.')

    if os.path.isdir(ground_truths):
        gt_files = sorted([x for x in os.listdir(ground_truths) if x.startswith(prefix) and x.endswith(suffix)])
        pred_files = sorted([x for x in os.listdir(predictions) if x.startswith(prefix) and x.endswith(suffix)])
        assert gt_files == pred_files, (f'GT files not found in predictions: {set(gt_files) - set(pred_files)}. '
                                        f'Prediction files not found in GT: {set(pred_files) - set(gt_files)}')
    else:
        gt_files = [ground_truths]
        pred_files = [predictions]

    assert output_path.endswith('.csv') or output_path.endswith('.json'), (f'Output path must be either .csv or .json, '
                                                                           f'is {output_path}')
    assert all([isinstance(x, int) for x in indices.values()]), f'Indices must be integers, found {indices.values()}.'
    print(f"Found {len(gt_files)} cases and {len(indices)} classes")

    metrics = aggregate_metrics(gt_files, pred_files, reorient, dtype, indices, num_workers)
    write_metrics(output_path, metrics, indices, console)


def dice(x: ndarray, y: ndarray) -> Tuple[int, int, int, float]:
    x_sum = x.sum()
    y_sum = y.sum()
    both = x_sum + y_sum
    common = 0

    if x_sum == y_sum == 0:
        score = 1.0
    elif x_sum == 0 or y_sum == 0:
        score = 0.0
    else:
        common = (x & y).sum()
        score = 2 * common / both

    return common, both, x_sum, score


def multi_class_dice(gt: ndarray, pred: ndarray, indices: Sequence[int]) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    common_voxels = []
    all_voxels = []
    gt_voxels = []
    dice_scores = []
    for label in indices:
        common, both, voxels, dice_score = dice(gt == label, pred == label)
        common_voxels.append(common)
        all_voxels.append(both)
        gt_voxels.append(voxels)
        dice_scores.append(dice_score)

    common_voxels = np.array(common_voxels)
    all_voxels = np.array(all_voxels)
    gt_voxels = np.array(gt_voxels)
    dice_scores = np.array(dice_scores)
    return common_voxels, all_voxels, gt_voxels, dice_scores


def evaluate_prediction(gt: str, pred: str, reorient: bool, dtype: np.dtype, indices: Sequence[int]) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    gt = read_mask(gt, reorient, dtype)
    pred = read_mask(pred, reorient, dtype)
    return multi_class_dice(gt, pred, indices)


def evaluate_predictions(gt_files: List[str], pred_files: List[str], reorient: bool, dtype: np.dtype,
                         indices: Sequence[int], num_workers: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    if num_workers == 0:
        ret = [evaluate_prediction(gt, pred, reorient, dtype, indices) for gt, pred in zip(gt_files, pred_files)]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            ret = executor.map(evaluate_prediction,
                               [(gt, pred, reorient, dtype, indices) for gt, pred in zip(gt_files, pred_files)])

    common_voxels = []
    all_voxels = []
    gt_voxels = []
    dice_scores = []
    for a, b, c, d in ret:
        common_voxels.append(a)
        all_voxels.append(b)
        gt_voxels.append(c)
        dice_scores.append(d)

    common_voxels = np.array(common_voxels)
    all_voxels = np.array(all_voxels)
    gt_voxels = np.array(gt_voxels)
    dice_scores = np.array(dice_scores)
    return common_voxels, all_voxels, gt_voxels, dice_scores


def aggregate_metrics(gt_files: List[str], pred_files: List[str], reorient: bool, dtype: np.dtype,
                      indices: dict, num_workers: int) -> dict:
    index_keys = indices.keys()
    index_values = indices.values()
    common_voxels, all_voxels, gt_voxels, dice_scores = evaluate_predictions(
        gt_files, pred_files, reorient, dtype, tuple(index_values), num_workers)
    metrics = {}
    for pred, scores, voxels in zip(pred_files, dice_scores, gt_voxels):
        pred = pred.split(os.path.sep)[-1]
        metrics[pred] = {}
        for score, label in zip(scores, index_keys):
            metrics[pred][label] = score
        metrics[pred]['Mean'] = np.mean(scores)
        metrics[pred]['Weighted mean'] = np.average(scores, weights=voxels)

    common_voxels = np.sum(common_voxels, axis=0)
    all_voxels = np.sum(all_voxels, axis=0)

    scores = np.mean(dice_scores, axis=0)
    metrics['Mean'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Mean']['Mean'] = np.mean(scores)
    metrics['Mean']['Weighted mean'] = np.average(scores, weights=np.sum(gt_voxels, axis=0))

    scores = np.average(dice_scores, axis=0, weights=np.sum(gt_voxels, axis=1))
    metrics['Weighted mean'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Weighted mean']['Mean'] = np.mean(scores)
    metrics['Weighted mean']['Weighted mean'] = np.average(scores, weights=np.sum(gt_voxels, axis=0))

    scores = []
    for common, both in zip(common_voxels, all_voxels):
        if both == 0:
            scores.append(1.0)
        else:
            scores.append(2 * common / both)
    metrics['Significant dice'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Significant dice']['Mean'] = np.mean(scores)
    metrics['Significant dice']['Weighted mean'] = np.average(scores, weights=np.sum(gt_voxels, axis=0))
    return metrics


def write_metrics(output_path: str, metrics: dict, indices: dict, console=bool):
    json_str = json.dumps(metrics, indent=2)
    if console:
        print(json_str)
    with open(output_path, 'w') as f:
        if output_path.endswith('.json'):
            f.write(json_str)
        else:
            columns = ['Cases', *indices.keys(), 'Mean', 'Weighted mean']
            f.write(','.join(columns) + '\n')
            for key, mapping in metrics.items():
                row = map(str, [key, *mapping.values()])
                f.write(','.join(row) + '\n')
