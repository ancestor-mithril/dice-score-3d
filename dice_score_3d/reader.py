import SimpleITK as sitk
import numpy as np
from numpy import ndarray


def read_mask(path: str, reorient: bool, dtype: np.dtype) -> ndarray:
    """ Reads a 3D volume using SimpleITK and returns the segmentation mask as a ndarray.
    Args:
        path (str): The path to the location of the segmentation mask.
        reorient (bool): If `True`, the segmentation mask is reoriented to the "LPS" orientation.
        dtype (np.dtype): The data type of the returned ndarray.
    """
    img = sitk.ReadImage(path)
    if reorient:
        img = sitk.DICOMOrient(img)
    return sitk.GetArrayFromImage(img).astype(dtype, copy=False)
