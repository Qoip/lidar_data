import imageio.v3 as iio
import numpy as np


def load_image(image_path):
    """Загрузка изображения LiDAR."""
    return iio.imread(image_path)


def prepare_3d_coordinates(image, angle_min=-30, angle_max=30):
    """Преобразование изображения LiDAR в координаты X, Y, Z."""
    h, w = image.shape
    angles = np.radians(np.linspace(angle_min, angle_max, w))
    times = np.arange(h) * 0.02

    angle_grid, time_grid = np.meshgrid(angles, times)

    r = image / 1000.0
    r[r > 20] = 0

    X = r * np.sin(angle_grid)
    Y = time_grid
    Z = -r * np.cos(angle_grid)

    return X, Y, Z
