import os
import tempfile
import unittest

import numpy as np

from dice_score_3d import dice_metrics
from dice_score_3d.metrics import dice, multi_class_dice, evaluate_prediction
from tests.utils import create_and_write_volume


class TestMetrics(unittest.TestCase):
    def test_dice(self):
        x: np.ndarray = np.random.random(size=(20, 21, 22)) > 0.5
        y: np.ndarray = np.random.random(size=(20, 21, 22)) > 0.5

        self.assertEqual(dice(x, x), (x.sum(), 2 * x.sum(), x.sum(), 1.0))
        self.assertEqual(dice(x, ~x), (0, x.size, x.sum(), 0.0))
        common, both, x_sum, score = dice(x, y)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_multi_class_dice(self):
        x = np.random.randint(0, 5, (20, 21, 22), dtype=np.uint8)
        y = np.random.randint(0, 5, (20, 21, 22), dtype=np.uint8)

        _, _, _, dice_scores = multi_class_dice(x, x, [1, 2, 3])
        self.assertTrue(all([x == 1.0 for x in dice_scores]))

        _, _, _, dice_scores = multi_class_dice(x, y, [1, 2, 3])
        self.assertTrue(all([0.0 <= x <= 1.0 for x in dice_scores]))

        _, _, _, dice_scores = multi_class_dice(x, y, [7, 8])
        self.assertTrue(all([x == 1.0 for x in dice_scores]))

    def test_evaluate_prediction(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        try:
            create_and_write_volume(tmp.name, random_direction=False)

            _, _, _, dice_scores = evaluate_prediction(tmp.name, tmp.name, False, np.uint8, [1, 2, 3])
            self.assertTrue(all([x == 1.0 for x in dice_scores]))
        finally:
            tmp.close()
            os.unlink(tmp.name)

    def test_dice_metrics(self):
        self.assertRaisesRegex(AssertionError, 'Prediction path and GT path must both be a single file or a folder',
                               dice_metrics, './', './random_string?.!@3$not_a_path', 'results.csv', {'Lung': 1})
        self.assertRaisesRegex(AssertionError, 'Output path must be either .csv or .json, is results.txt',
                               dice_metrics, './', './', 'results.txt', {'Lung': 1})
        self.assertRaisesRegex(AssertionError, 'Indices must be integers, found .*',
                               dice_metrics, './', './', 'results.csv', {'Lung': 'text'})


if __name__ == '__main__':
    unittest.main()
