import os
import tempfile
import unittest

import numpy as np

from dice_score_3d.readers import read_sitk, read_nibabel, read_mask
from tests.utils import create_and_write_volume


class TestReaders(unittest.TestCase):
    def test_readers_are_equivalent(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        try:
            create_and_write_volume(tmp.name, random_direction=False)
            sitk_array = read_sitk(tmp.name, reorient=False)
            nibabel_array = read_nibabel(tmp.name, reorient=False)
            self.assertEqual(sitk_array.shape, nibabel_array.shape)
            self.assertTrue(np.array_equal(sitk_array, nibabel_array))
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
            read_sitk(tmp.name, reorient=True)
            read_nibabel(tmp.name, reorient=True)  # creates warning because the direction matrix is not orthogonal
        finally:
            tmp.close()
            os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
