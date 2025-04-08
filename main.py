import os
import json
from src.image_utils import load_image, prepare_3d_coordinates
from src.truck_bed_detection import detect_truck_bed
from scipy.spatial import ConvexHull
import numpy as np


def save_text_results(output_dir, category, filename, plane_params, inlier_points):
    """Запись текстовых результатов в JSON файл для каждого изображения в категории."""
    category_dir = os.path.join(output_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    result_file_path = os.path.join(category_dir, filename.replace('.png', '.json'))

    if isinstance(plane_params, tuple):
        plane_params = [float(param.item()) if isinstance(param, np.ndarray) else float(param)
                        for param in plane_params]

    if isinstance(inlier_points, np.ndarray):
        inlier_points = inlier_points.tolist()

    result_data = {
        "filename": filename,
        "plane_params": plane_params,
        "inlier_points": inlier_points if inlier_points is not None else None,
        "category": category
    }

    with open(result_file_path, "w") as f:
        json.dump(result_data, f, indent=4)


def process_directory(input_dir, output_dir):
    """Обработка всех изображений в директории."""
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)

        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.png'):
                    image_path = os.path.join(category_path, filename)
                    image = load_image(image_path)

                    X, Y, Z = prepare_3d_coordinates(image)
                    plane_params, inlier_points = detect_truck_bed(X, Y, Z)
                    # print("I am here", image_path, image.shape, X.shape, Y.shape, Z.shape)

                    inlier_mask = ~np.isnan(inlier_points[:, 0])
                    inlier_X = inlier_points[inlier_mask, 0]
                    inlier_Y = inlier_points[inlier_mask, 1]
                    inlier_Z = inlier_points[inlier_mask, 2]

                    points_2d = np.column_stack((inlier_X, inlier_Y))

                    hull = ConvexHull(points_2d)

                    hull_x = inlier_X[hull.vertices]
                    hull_y = inlier_Y[hull.vertices]
                    hull_z = inlier_Z[hull.vertices]

                    final_points = np.column_stack((hull_x, hull_y, hull_z))

                    save_text_results(output_dir, category, filename, plane_params, final_points)

                    print(f"Обработано {filename}, параметры плоскости: {plane_params}, "
                          f"количество точек: {len(final_points)}")


if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "data/results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_dir, output_dir)
