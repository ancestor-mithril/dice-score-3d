import json
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence, Tuple, Union

import numpy as np
from dice_score_3d.reader import read_mask
from numpy import ndarray


def dice_metrics(ground_truths: str, predictions: str, output_path: str, indices: dict, reorient: bool = False,
                 dtype: str = 'uint8', prefix: str = '', suffix: str = '.nii.gz', num_workers: int = 0,
                 console: bool = False):
    """ Calculates Dice metrics for pairs of predictions and GT, writing the aggregated results in a csv or json file.

    Args:
        ground_truths (str): Path to Ground Truth. Can be a single file or a folder with all the GT volumes. The number
            of GT files must match the number of predictions.When passing a folder of GT files, the name of the GT
            files must match the name of the predictions. This is not applicable when passing a single file. Supported
            file formats: .nii, .nii.gz, .nrrd, .mha, .gipl.
        predictions (str): Path to Ground Truth. Can be a single file or a folder with all the predicted volumes. The
            number of prediction files must match the number of GT files. When passing a folder of prediction files,
            the name of the prediction files must match the name of the GT files. This is not applicable when passing a
            single file. Supported file formats: .nii, .nii.gz, .nrrd, .mha, .gipl.
        output_path (str): The output path to write the computed metrics. Can be a csv or json file, depending on
            extension. Example: "results.csv", "results.json".
        indices (dict): Dictionary describing the indices used for calculating the Dice Similarity Coefficient.
            Example: `{"lung_left": 1, "lung_right": 2}`.
        reorient (bool): If `True`, reorients both the GT and the prediction to the default "LPS" orientation before
            calculating the Dice Score. Default: `False`.
        dtype (str): Must be either "uint8" when having less than 255 classes, or "uint16" otherwise.
            Default: `'uint8'`.
        prefix (str): This parameter is used when the ground truth path is a folder. It filters all the files in the
            folder and selects only the files with this prefix. Default: `''`
        suffix (str): This parameter is used when the ground truth path is a folder. It filters all the files in the
            folder and selects only the files with this suffix. Default: `'.nii.gz'`.
        num_workers (int): Number of parallel processes to be used to calculate the Dice Score in parallel. Default:
            `0`.
        console (bool): If `True`, also prints the Dice metrics to console. Default: `False`.
    """
    dtype = np.uint8 if dtype == 'uint8' else np.uint16
    assert os.path.isfile(ground_truths) and os.path.isfile(predictions) or \
           os.path.isdir(ground_truths) and os.path.isdir(predictions), ('Prediction path and GT path must both be a '
                                                                         'single file or a folder.')

    if os.path.isdir(ground_truths):
        gt_files = sorted([x for x in os.listdir(ground_truths) if x.startswith(prefix) and x.endswith(suffix)])
        pred_files = sorted([x for x in os.listdir(predictions) if x.startswith(prefix) and x.endswith(suffix)])
        assert gt_files == pred_files, (f'GT files not found in predictions: {set(gt_files) - set(pred_files)}. '
                                        f'Prediction files not found in GT: {set(pred_files) - set(gt_files)}')
        gt_files = [os.path.join(ground_truths, x) for x in gt_files]
        pred_files = [os.path.join(predictions, x) for x in pred_files]
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
    """ Calculates the Dice Score and collects common, GT and the union of voxels for a pair of prediction and GT
    using a single index (label).
    """
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
    """ Calculates the Dice Score and collects common, GT and the union of voxels for a pair of prediction and GT
    using all indices (labels).
    """
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
    """ Evaluates a single pair of prediction and GT and collects metrics.
    """
    gt = read_mask(gt, reorient, dtype)
    pred = read_mask(pred, reorient, dtype)
    return multi_class_dice(gt, pred, indices)


def evaluate_predictions(gt_files: List[str], pred_files: List[str], reorient: bool, dtype: np.dtype,
                         indices: Sequence[int], num_workers: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Evaluates each pair of prediction and GT sequentially or in parallel and collects metrics.
    """
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


def average(x: Union[ndarray, Sequence], axis: int = None, weights: ndarray = None) -> ndarray:
    if np.all(weights == 0):
        return np.mean(x, axis=axis)
    return np.average(x, axis=axis, weights=weights)


def aggregate_metrics(gt_files: List[str], pred_files: List[str], reorient: bool, dtype: np.dtype,
                      indices: dict, num_workers: int) -> dict:
    """ Evaluates and aggregates metrics from each pair of prediction and GT, calculating the Dice Score for each label,
    the mean and weighted mean for each case and also the per-label mean, weighted mean and Union Dice. The Union Dice
    is calculated as if all volumes are combined into one single volume.
    """
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
        metrics[pred]['Weighted mean'] = average(scores, weights=voxels)

    common_voxels = np.sum(common_voxels, axis=0)
    all_voxels = np.sum(all_voxels, axis=0)

    scores = np.mean(dice_scores, axis=0)
    metrics['Mean'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Mean']['Mean'] = np.mean(scores)
    metrics['Mean']['Weighted mean'] = average(scores, weights=np.sum(gt_voxels, axis=0))

    scores = average(dice_scores, axis=0, weights=np.sum(gt_voxels, axis=1))
    metrics['Weighted mean'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Weighted mean']['Mean'] = np.mean(scores)
    metrics['Weighted mean']['Weighted mean'] = average(scores, weights=np.sum(gt_voxels, axis=0))

    scores = []
    for common, both in zip(common_voxels, all_voxels):
        if both == 0:
            scores.append(1.0)
        else:
            scores.append(2 * common / both)
    metrics['Union dice'] = {label: score for label, score in zip(index_keys, scores)}
    metrics['Union dice']['Mean'] = np.mean(scores)
    metrics['Union dice']['Weighted mean'] = average(scores, weights=np.sum(gt_voxels, axis=0))
    return metrics


def write_metrics(output_path: str, metrics: dict, indices: dict, console=bool):
    """ Writes the metrics to the csv or json file. Also prints to console if `console` is `True`.
    """
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
