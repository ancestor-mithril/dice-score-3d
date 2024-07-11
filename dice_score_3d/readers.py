import SimpleITK as sitk
import nibabel
import numpy as np
from nibabel import io_orientation


class ReadError(Exception):
    pass


def read_sitk(path: str, reorient: bool):
    try:
        img = sitk.ReadImage(path)
    except RuntimeError as e:
        # "ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found!"
        # Sometimes SimpleITK just refuses to read invalid volumes. In this case, we try to use nibabel
        raise ReadError('SimpleITK reading error') from e
    if reorient:
        img = sitk.DICOMOrient(img)
    return sitk.GetArrayFromImage(img)


def read_nibabel(path: str, reorient: bool):
    img = nibabel.load(path)
    if reorient:
        img = img.as_reoriented(io_orientation(img.affine))
    return img.get_fdata()


def robust_read(path: str, reorient: bool):
    try:
        return read_sitk(path, reorient)
    except ReadError as e1:
        try:  # We try to read with nibabel
            return read_nibabel(path, reorient)
        except Exception as e2:
            raise e2 from e1  # we raise both exceptions


def read_mask(path: str, reorient: bool, dtype: np.dtype):
    return robust_read(path, reorient).astype(dtype, copy=False)
