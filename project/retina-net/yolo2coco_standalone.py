import os
import json
import glob
from tqdm import tqdm
from PIL import Image

def convert_yolo_to_coco(images_dir, labels_dir, class_list, output_path):
    with open(class_list, 'r') as f:
        categories = [{"id": i, "name": name.strip()} for i, name in enumerate(f.readlines())]

    image_id = 0
    annotation_id = 0
    images = []
    annotations = []

    for image_file in tqdm(glob.glob(os.path.join(images_dir, '*'))):
        if not image_file.lower().endswith(('jpg', 'jpeg', 'png')):
            continue

        filename = os.path.basename(image_file)
        label_file = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')

        if not os.path.exists(label_file):
            continue

        img = Image.open(image_file)
        width, height = img.size

        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename,
        })

        with open(label_file, 'r') as lf:
            for line in lf:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                box_width = w * width
                box_height = h * height

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                })
                annotation_id += 1

        image_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"Saved COCO file to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to labels directory")
    parser.add_argument("--classes", required=True, help="Path to classes.txt")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    convert_yolo_to_coco(
        images_dir=args.images,
        labels_dir=args.labels,
        class_list=args.classes,
        output_path=args.output
    )

'''
Convert training set:
python3 yolo2coco_standalone.py \
  --images train/images \
  --labels train/labels \
  --classes classes.txt \
  --output annotations/instances_train.json

Convert validation set:
python3 yolo2coco_standalone.py \
  --images valid/images \
  --labels valid/labels \
  --classes classes.txt \
  --output annotations/instances_val.json

Convert test set:
python3 yolo2coco_standalone.py \
  --images test/images \
  --labels test/labels \
  --classes classes.txt \
  --output annotations/instances_test.json
'''