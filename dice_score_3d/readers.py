import SimpleITK as sitk
import nibabel as nib
import numpy as np


def read_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def read_nibabel(path):
    return nib.load(path).get_fdata()


def robust_read(path):
    try:
        return read_sitk(path)
    except RuntimeError:
        # ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found!
        pass
    return read_nibabel(path)


def read_mask(path):
    mask = robust_read(path)
    if mask.dtype in (np.uint8, np.uint16, np.int8, np.int16):
        return mask
    return mask.astype(np.uint8 if np.max(mask) < 255 else np.uint16)
