import os
import tempfile
import unittest

import numpy as np

from dice_score_3d.reader import read_mask
from tests.utils import create_and_write_volume


class TestReader(unittest.TestCase):
    def test_default_orientation(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        try:
            create_and_write_volume(tmp.name, random_direction=False)
            array_1 = read_mask(tmp.name, reorient=False, dtype=np.uint8)
            array_2 = read_mask(tmp.name, reorient=True, dtype=np.uint8)
            self.assertTrue(np.array_equal(array_1, array_2))
        finally:
            tmp.close()
            os.unlink(tmp.name)

    def test_supported_extensions(self):
        def do_read(extension: str):
            tmp = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
            try:
                # Default orientation
                create_and_write_volume(tmp.name, random_direction=False)
                read_mask(tmp.name, reorient=False, dtype=np.uint8)
            finally:
                tmp.close()
                os.unlink(tmp.name)

        do_read('.nii.gz')
        do_read('.nii')
        do_read('.nrrd')
        do_read('.mha')
        do_read('.gipl')
        self.assertRaises(Exception, do_read, '.txt')

    def test_reorientation(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        try:
            create_and_write_volume(tmp.name, random_direction=True)
            array_1 = read_mask(tmp.name, reorient=False, dtype=np.uint8)
            array_2 = read_mask(tmp.name, reorient=True, dtype=np.uint8)
            self.assertFalse(np.array_equal(array_1, array_2))
        finally:
            tmp.close()
            os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
