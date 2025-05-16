# python3 convert_yolo_polygon_to_coco.py

import os
import json
import cv2
from glob import glob
from tqdm import tqdm

def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_coco_json(image_dir, label_dir, classes_path, output_path):
    class_names = load_classes(classes_path)
    class_name_to_id = {name: i for i, name in enumerate(class_names)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    ann_id = 0

    for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing")):
        filename = os.path.basename(img_path)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        # Get image size
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7 or len(parts[1:]) % 2 != 0:
                    continue  # skip invalid polygons

                class_id = int(float(parts[0]))
                polygon = list(map(float, parts[1:]))

                xs = polygon[::2]
                ys = polygon[1::2]
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)
                bbox = [x_min * width, y_min * height, 
                        (x_max - x_min) * width, (y_max - y_min) * height]

                segmentation = [[x * width if i % 2 == 0 else x * height for i, x in enumerate(polygon)]]

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": segmentation,
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f)
    print(f"Saved COCO JSON to {output_path}")

# ==== Edit paths below ====
create_coco_json(
    image_dir="valid/images",
    label_dir="valid/labels",
    classes_path="classes.txt",
    output_path="annotations/instances_valid.json"
)

'''
create_coco_json(
    image_dir="train/images",
    label_dir="train/labels",
    classes_path="classes.txt",
    output_path="annotations/instances_train.json"
)
create_coco_json(
    image_dir="test/images",
    label_dir="test/labels",
    classes_path="classes.txt",
    output_path="annotations/instances_test.json"
)
create_coco_json(
    image_dir="valid/images",
    label_dir="valid/labels",
    classes_path="classes.txt",
    output_path="annotations/instances_valid.json"
)
'''
