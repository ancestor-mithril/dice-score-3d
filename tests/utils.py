import numpy as np
import SimpleITK as sitk


def create_random_volume(high=5, size=(22, 21, 20), random_direction=False, dtype=np.uint8):
    x = np.random.randint(low=0, high=high, size=size, dtype=dtype)
    spacing = (1, 1, 1)
    origin = (0, 0, 0)
    if random_direction:
        direction = (-1, *np.random.uniform(low=-1.0, high=1.0, size=(8,)))
    else:
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return x, spacing, origin, direction


def write_volume(path, volume, spacing, origin, direction):
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)


def create_and_write_volume(path, high=5, size=(22, 21, 20), random_direction=False, dtype=np.uint8):
    volume, spacing, origin, direction = create_random_volume(high, size, random_direction, dtype=dtype)
    write_volume(path, volume, spacing, origin, direction)
