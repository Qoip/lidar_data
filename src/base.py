from typing import Optional, Tuple
import numpy as np


class PlaneDetectionResult:
    def __init__(
        self,
        plane_coeffs: Tuple[float, float, float, float],
        bottom_points: Optional[np.ndarray] = None,
        bottom_hull: Optional[np.ndarray] = None,
        leftside_points: Optional[np.ndarray] = None,
        rightside_points: Optional[np.ndarray] = None,
        front_points: Optional[np.ndarray] = None,
        back_points: Optional[np.ndarray] = None,
    ):
        self.plane_coeffs = plane_coeffs
        self.bottom_points = bottom_points
        self.bottom_hull = bottom_hull
        self.leftside_points = leftside_points
        self.rightside_points = rightside_points
        self.front_points = front_points
        self.back_points = back_points
