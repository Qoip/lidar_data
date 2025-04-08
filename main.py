import os
import json
from src.image_utils import load_image, preprocess_image, prepare_3d_coordinates
from src.truck_bed_detection import detect_truck_bed


def save_text_results(output_dir, category, filename, plane_params, inlier_points):
    """Запись текстовых результатов в JSON файл для каждого изображения в категории."""
    category_dir = os.path.join(output_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    result_file_path = os.path.join(category_dir, filename.replace('.png', '.json'))

    result_data = {
        "filename": filename,
        "plane_params": plane_params,
        "inlier_points": inlier_points.tolist() if inlier_points is not None else None,
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

                    image = preprocess_image(image)

                    plane_params, inlier_points = detect_truck_bed(*prepare_3d_coordinates(image))

                    save_text_results(output_dir, category, filename, plane_params, inlier_points)

                    print(f"Обработано {filename}, параметры плоскости: {plane_params}")


if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "data/results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_dir, output_dir)
