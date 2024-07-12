import os
import tempfile
import unittest
import SimpleITK as sitk
import numpy as np
from dice_score_3d.readers import read_sitk, read_nibabel


def create_random_volume(high=5, size=(20, 20, 20), random_direction=False):
    x = np.random.randint(low=0, high=high, size=size, dtype=np.uint8 if high < 255 else np.uint16)
    spacing = (1, 1, 1)
    origin = (0, 0, 0)
    if random_direction:
        direction = tuple(np.random.uniform(low=-1.0, high=1.0, size=(9,)))
    else:
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return x, spacing, origin, direction


def write_volume(path, volume, spacing, origin, direction):
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)


def create_and_write_volume(path, high=5, size=(22, 21, 20), random_direction=False):
    volume, spacing, origin, direction = create_random_volume(high, size, random_direction)
    write_volume(path, volume, spacing, origin, direction)


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
