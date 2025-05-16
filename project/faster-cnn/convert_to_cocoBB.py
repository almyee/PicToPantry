# import torch
# from torch.utils.data import Dataset

# class CocoSegToBBoxDataset(Dataset):
#     def __init__(self, coco, image_ids, transforms=None):
#         """
#         coco: COCO object from pycocotools.coco.COCO
#         image_ids: list of image ids to load
#         transforms: optional image and target transforms
#         """
#         self.coco = coco
#         self.image_ids = image_ids
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.image_ids)

#     def polygon_to_bbox(self, segmentation):
#         all_x = []
#         all_y = []
#         # segmentation is list of polygons (each polygon is a list of coords)
#         for polygon in segmentation:
#             xs = polygon[0::2]
#             ys = polygon[1::2]
#             all_x.extend(xs)
#             all_y.extend(ys)
#         xmin = min(all_x)
#         xmax = max(all_x)
#         ymin = min(all_y)
#         ymax = max(all_y)
#         return [xmin, ymin, xmax, ymax]

#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         img_info = self.coco.loadImgs(image_id)[0]
#         path = img_info['file_name']

#         # Load image (using PIL or cv2)
#         from PIL import Image
#         image = Image.open(path).convert("RGB")

#         # Get all annotation ids for this image
#         ann_ids = self.coco.getAnnIds(imgIds=image_id)
#         anns = self.coco.loadAnns(ann_ids)

#         boxes = []
#         labels = []
#         masks = []

#         for ann in anns:
#             # convert segmentation polygons to bounding box
#             bbox = self.polygon_to_bbox(ann['segmentation'])

#             # sanity check for bbox width and height > 0
#             xmin, ymin, xmax, ymax = bbox
#             if xmax <= xmin or ymax <= ymin:
#                 # skip invalid bbox
#                 continue

#             boxes.append(bbox)
#             labels.append(ann['category_id'])

#             # optionally, you can process masks too (if needed)
#             # mask = self.coco.annToMask(ann)
#             # masks.append(torch.as_tensor(mask, dtype=torch.uint8))

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         # If you want to use masks uncomment below:
#         # masks = torch.stack(masks) if masks else torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["image_id"] = torch.tensor([image_id])

#         # Add masks if you want:
#         # target["masks"] = masks

#         if self.transforms:
#             image, target = self.transforms(image, target)

#         return image, target

# from pycocotools.coco import COCO

# coco = COCO('Food-Recognition-1/annotations/instances_train.json')
# image_ids = coco.getImgIds()

# dataset = CocoSegToBBoxDataset(coco, image_ids)

# image, target = dataset[0]

# print(target["boxes"])  # bounding boxes in [xmin, ymin, xmax, ymax] format
# print(target["labels"]) # category labels

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np

class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root  # folder with images
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        filename = img_info['file_name']

        # Full image path
        img_path = os.path.join(self.root, filename)
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Convert segmentation polygons to masks (if you need masks)
            if 'segmentation' in ann:
                seg = ann['segmentation']
                # You can convert polygons to masks here if needed (optional)
            
            # COCO bounding box format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert bbox to [xmin, ymin, xmax, ymax]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            # masks.append(...)  # Optional if you want masks

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)

train_dataset = CocoDataset(
    root='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/images/valid',
    annotation_file='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/annotations/instances_valid.json',
    transforms=None  
)
'''
train_dataset = CocoDataset(
    root='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/images/train',
    annotation_file='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/annotations/instances_train.json',
    transforms=None  # or your transforms function
)
train_dataset = CocoDataset(
    root='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/images/test',
    annotation_file='/root/ecs271_files/PicToPantry/project/faster-cnn/Food-Recognition-1/annotations/instances_test.json',
    transforms=None  
)
'''