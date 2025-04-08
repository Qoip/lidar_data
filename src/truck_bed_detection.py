import numpy as np
from sklearn.linear_model import RANSACRegressor


def detect_truck_bed(X, Y, Z):
    """Ищет плоскость дна кузова по самому плотному слою в центре сцены."""

    mask = ~np.isnan(Z)
    Z_valid = Z[mask]
    X_valid = X[mask]
    # Y_valid = Y[mask]
    flat_indices = np.flatnonzero(mask)

    # Группировка Z по слоям
    rounded_z = np.round(Z_valid / 0.1) * 0.1
    unique_z = np.unique(rounded_z)

    x_min, x_max = np.min(X_valid), np.max(X_valid)
    center_left = x_min + (x_max - x_min) / 3
    center_right = x_max - (x_max - x_min) / 3

    best_z = None
    best_score = -np.inf

    for z_val in unique_z:
        layer_mask = np.abs(Z_valid - z_val) < 0.1
        X_layer = X_valid[layer_mask]

        center_count = np.sum((X_layer >= center_left) & (X_layer <= center_right))
        side_count = np.sum((X_layer < center_left) | (X_layer > center_right))

        if center_count > side_count and center_count > best_score:
            best_score = center_count
            best_z = z_val

    if best_z is None:
        raise RuntimeError("Не удалось найти подходящий слой")

    # Отбор точек слоя
    close_mask = np.abs(Z_valid - best_z) < 0.1
    Z_sel = Z_valid[close_mask].reshape(-1, 1)
    dummy_X = np.ones_like(Z_sel)

    model = RANSACRegressor()
    model.fit(dummy_X, Z_sel)

    Z_plane = model.predict([[1]])[0]
    A, B, C = 0, 0, 1
    D = -Z_plane

    inliers_local = model.inlier_mask_
    selected_indices = flat_indices[close_mask]
    inlier_flat_indices = selected_indices[inliers_local]

    # Преобразуем flat индексы обратно в 2D
    inlier_2d_indices = np.unravel_index(inlier_flat_indices, Z.shape)

    final_mask = np.zeros_like(Z, dtype=bool)
    final_mask[inlier_2d_indices] = True

    inlier_points = np.column_stack((X[final_mask], Y[final_mask], Z[final_mask]))
    return (A, B, C, D), inlier_points
