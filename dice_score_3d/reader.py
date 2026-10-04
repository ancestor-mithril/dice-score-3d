import SimpleITK as sitk
import numpy as np
from numpy import ndarray


def read_mask_nibabel(path: str, reorient: bool, dtype: np.dtype) -> ndarray:
    """
    Reads a 3D volume using NiBabel and returns the segmentation mask
    as a NumPy ndarray.

    Args:
        path: Path to the segmentation mask.
        reorient: If True, reorient the mask to LPS orientation.
        dtype: Data type of the returned array.

    Returns:
        Segmentation mask with shape (z, y, x), matching SimpleITK.
    """
    import nibabel as nib
    try:
        nii = nib.load(path)
        data = np.asanyarray(nii.dataobj)

        if reorient:
            current_orientation = nib.orientations.io_orientation(nii.affine)
            target_orientation = nib.orientations.axcodes2ornt(("L", "P", "S"))

            transform = nib.orientations.ornt_transform(
                current_orientation,
                target_orientation,
            )
            data = nib.orientations.apply_orientation(data, transform)

        # NiBabel:   (x, y, z)
        # SimpleITK: (z, y, x)
        data = data.transpose(2, 1, 0)

        return data.astype(dtype, copy=False)

    except Exception as e:
        print(f"Failed reading {path} due to {e}")
        raise


def read_mask(path: str, reorient: bool, dtype: np.dtype) -> ndarray:
    """ Reads a 3D volume using SimpleITK and returns the segmentation mask as a ndarray.
    Args:
        path (str): The path to the location of the segmentation mask.
        reorient (bool): If `True`, the segmentation mask is reoriented to the "LPS" orientation.
        dtype (np.dtype): The data type of the returned ndarray.
    """
    try:
        img = sitk.ReadImage(path)
        if reorient:
            img = sitk.DICOMOrient(img)
        return sitk.GetArrayFromImage(img).astype(dtype, copy=False)
    except Exception as e:
        return read_mask_nibabel(path, reorient, dtype)
