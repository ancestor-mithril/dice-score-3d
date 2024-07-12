import os
import tempfile
import unittest

import numpy as np

from dice_score_3d.readers import read_sitk, read_nibabel
from tests.utils import create_and_write_volume


class TestReaders(unittest.TestCase):
    def test_readers_are_equivalent(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        try:
            create_and_write_volume(tmp.name)
            sitk_array = read_sitk(tmp.name, reorient=False)
            nibabel_array = read_nibabel(tmp.name, reorient=False)
            self.assertEqual(sitk_array.shape, nibabel_array.shape)
            self.assertTrue(np.array_equal(sitk_array, nibabel_array))
        finally:
            tmp.close()
            os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
