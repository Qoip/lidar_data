import numpy as np
from sklearn.linear_model import RANSACRegressor


def detect_truck_bed(X, Y, Z):
    """Функция для поиска дна кузова с использованием RANSAC, возвращающая плоскость и многоугольник."""

    mask = np.isnan(Z)
    X, Y, Z = X[~mask], Y[~mask], Z[~mask]

    model = RANSACRegressor()
    model.fit(np.column_stack((X, Y)), Z)

    A, B = model.estimator_.coef_
    C = -1
    D = -model.estimator_.intercept_

    inlier_mask = model.inlier_mask_
    inlier_points = np.column_stack((X[inlier_mask], Y[inlier_mask], Z[inlier_mask]))

    return (A, B, C, D), inlier_points
