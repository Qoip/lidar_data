import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from base import PlaneDetectionResult
from image_utils import prepare_3d_coordinates

DEBUG_PLOT = False


def detect(img) -> PlaneDetectionResult:
    """Ищет плоскость дна кузова по самому плотному слою в центре сцены."""

    X, Y, Z = prepare_3d_coordinates(img)
    mask = ~np.isnan(Z)
    Z_valid = Z[mask]
    X_valid = X[mask]
    Y_valid = Y[mask]

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

    close_mask = np.abs(Z_valid - best_z) < 0.1
    Z_sel = Z_valid[close_mask].reshape(-1, 1)
    dummy_X = np.ones_like(Z_sel)

    model = RANSACRegressor()
    model.fit(dummy_X, Z_sel)

    Z_plane = model.predict([[1]])[0]
    A, B, C = 0, 0, 1
    D = -Z_plane.item()

    close_mask = np.abs(Z_valid - best_z) < 0.1
    X_sel = X_valid[close_mask]
    Y_sel = Y_valid[close_mask]
    Z_sel = Z_valid[close_mask]

    all_points = np.column_stack((X_valid, Y_valid))
    tree = cKDTree(all_points)

    valid_points = []

    for x, y, z in zip(X_sel, Y_sel, Z_sel):
        neighbors = tree.query_ball_point([x, y], r=0.2)
        neighbors_points = np.array(
            [X_valid[i] for i in neighbors]), np.array(
            [Y_valid[i] for i in neighbors]), np.array(
            [Z_valid[i] for i in neighbors])

        neighbor_z = neighbors_points[2]
        count_above = np.sum(neighbor_z > z + 0.5)

        if count_above >= 5:
            valid_points.append([x, y, z])
        # print("X, Y, Z:", x, y, z)
        # print("Neighbors:", neighbors_points)

    valid_points = np.array(valid_points)

    if len(valid_points) > 0:
        db = DBSCAN(eps=0.1, min_samples=10)
        labels = db.fit_predict(valid_points[:, :2])

        unique_labels, counts = np.unique(labels, return_counts=True)

        valid_labels = unique_labels[unique_labels != -1]
        valid_counts = counts[unique_labels != -1]

        sorted_labels_by_size = sorted(zip(valid_labels, valid_counts), key=lambda x: x[1], reverse=True)

        top1_label, top1_size = sorted_labels_by_size[0]
        top2_label, top2_size = sorted_labels_by_size[1] if len(sorted_labels_by_size) > 1 else (None, 0)

        if top2_size >= top1_size / 2:
            top1_points = valid_points[labels == top1_label]
            top2_points = valid_points[labels == top2_label] if top2_label is not None else np.array([])

            final_cluster = np.vstack([top1_points, top2_points])
        else:
            final_cluster = valid_points[labels == top1_label]

        if DEBUG_PLOT:
            print("Количество точек в итоговом кластере:", len(final_cluster))
            if len(final_cluster) > 0:
                plt.figure(figsize=(6, 6))
                plt.scatter(final_cluster[:, 0], final_cluster[:, 1], s=2, c="blue")
                plt.title("Итоговый кластер (с объединением)")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.axis("equal")
                plt.grid(True)
                plt.show()
            else:
                print("Нет точек, удовлетворяющих условию.")

    margin = 0.5
    mask_valid_center = (X_valid >= center_left + margin) & (X_valid <= center_right - margin)
    X_center = X_valid[mask_valid_center & close_mask]
    Y_center = Y_valid[mask_valid_center & close_mask]
    Z_center = Z_valid[mask_valid_center & close_mask]

    if len(X_center) > 0:
        try:
            model = RANSACRegressor(estimator=LinearRegression())
            model.fit(np.column_stack((X_center, Y_center)), Z_center)

            if model.estimator_ is not None:
                A, B = model.estimator_.coef_
                D = model.estimator_.intercept_
                C = -1
            else:
                A, B, C, D = 0, 0, 1, -Z_plane.item()
        except Exception as e:
            print(f"RANSAC error: {str(e)}")
            A, B, C, D = 0, 0, 1, -Z_plane.item()
    else:
        A, B, C, D = 0, 0, 1, -Z_plane.item()

    hull = ConvexHull(final_cluster[:, :2])
    hull_x = final_cluster[hull.vertices, 0]
    hull_y = final_cluster[hull.vertices, 1]
    hull_z = final_cluster[hull.vertices, 2]
    return PlaneDetectionResult(
        plane_coeffs=(A, B, C, D),
        bottom_hull=np.column_stack((hull_x, hull_y, hull_z)),
    )
