import os
import json
import numpy as np
from src.image_utils import load_image, prepare_3d_coordinates
from src.solutions import get_solution_class


def save_text_results(output_dir, category, filename, plane_result):
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    result_file_path = os.path.join(category_dir, filename.replace('.png', '.json'))

    if isinstance(plane_result.plane_coeffs, tuple):
        plane_params = [float(p) for p in plane_result.plane_coeffs]
    else:
        plane_params = None

    if isinstance(plane_result.bottom_hull, np.ndarray):
        bottom_hull = plane_result.bottom_hull.tolist()
    else:
        bottom_hull = None

    result_data = {
        "filename": filename,
        "plane_params": plane_params,
        "inlier_points": bottom_hull,
        "category": category
    }

    with open(result_file_path, "w") as f:
        json.dump(result_data, f, indent=4)


def process_directory(input_dir, output_dir, solution_class):
    detector = solution_class()

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if not filename.endswith('.png'):
                continue

            image_path = os.path.join(category_path, filename)
            image = load_image(image_path)

            X, Y, Z = prepare_3d_coordinates(image)
            result = detector.detect(X, Y, Z)

            save_text_results(output_dir, category, filename, result)

            print(f"Обработано {filename}, параметры плоскости: {result.plane_coeffs}, "
                  f"количество точек: {len(result.bottom_hull) if result.bottom_hull is not None else 0}")


if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)

    print("Доступные методы:")
    for i, name in enumerate(get_solution_class.available_solutions(), start=1):
        print(f"{i}. {name}")

    idx = int(input("Выберите номер метода: "))
    selected_class = get_solution_class.by_index(idx)

    process_directory(input_dir, output_dir, selected_class)
