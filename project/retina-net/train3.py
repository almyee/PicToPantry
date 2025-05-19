# currently used convert_to_cocoBB.py and this file to train, so far no error but still running
# import torch
# from torch.utils.data import DataLoader
# import torchvision
# from PIL import Image
# import os
# import json
# from tqdm import tqdm  # add at the top
# from torch.utils.data import Subset


# class CocoDataset(torch.utils.data.Dataset):
#     def __init__(self, root, annotation_file, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         with open(annotation_file) as f:
#             self.coco = json.load(f)

#         self.image_info = {img['id']: img for img in self.coco['images']}
#         self.annotations = {}
#         for ann in self.coco['annotations']:
#             img_id = ann['image_id']
#             self.annotations.setdefault(img_id, []).append(ann)

#         # Filter only images with at least one annotation
#         self.ids = [img_id for img_id in self.image_info.keys()
#                     if img_id in self.annotations and len(self.annotations[img_id]) > 0]

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         img_info = self.image_info[img_id]
#         img_path = os.path.join(self.root, img_info['file_name'])
#         img = Image.open(img_path).convert("RGB")

#         annots = self.annotations.get(img_id, [])
#         boxes = []
#         labels = []
#         for ann in annots:
#             x, y, w, h = ann['bbox']
#             if w <= 0 or h <= 0:
#                 continue
#             boxes.append([x, y, x + w, y + h])
#             labels.append(ann['category_id'])

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         target = {
#             "boxes": boxes,
#             "labels": labels,
#             "image_id": torch.tensor([img_id])
#         }

#         if self.transforms:
#             img, target = self.transforms(img, target)

#         return img, target

# def get_transform():
#     import torchvision.transforms as T
#     def transform(image, target):
#         image = T.ToTensor()(image)
#         return image, target
#     return transform

# def collate_fn(batch):
#     return tuple(zip(*batch))

# def main():
#     # === Modify your dataset path here ===
#     dataset_root = "Food-Recognition-1/train/images"
#     annotation_file = "Food-Recognition-1/annotations/instances_train.json"

#     num_classes = 498  # including background

#     dataset = CocoDataset(dataset_root, annotation_file, transforms=get_transform())
#     dataset = Subset(dataset, range(200))  # Try with 200 samples first
#     # data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#     # data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
#     data_loader = DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4,  # or 8 depending on CPU cores
#         pin_memory=True
#     )


#     # === Load RetinaNet ===
#     # model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
#     # model.head.classification_head.num_classes = num_classes
#     from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
#     from torchvision.models.detection.retinanet import RetinaNetClassificationHead

#     # Load pre-trained model with weights
#     weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
#     model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)

#     # Get number of anchors
#     num_anchors = model.head.classification_head.num_anchors

#     # Replace classification head with correct num_classes
#     in_channels = model.backbone.out_channels
#     model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model.to(device)

#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#     num_epochs = 10

#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
#         print(f"\n[Epoch {epoch + 1}/{num_epochs}] Starting...", flush=True)

#         for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
#             try:
#                 images = [img.to(device) for img in images]
#                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#                 loss_dict = model(images, targets)
#                 losses = sum(loss for loss in loss_dict.values())

#                 optimizer.zero_grad()
#                 losses.backward()
#                 optimizer.step()

#                 epoch_loss += losses.item()

#                 if (batch_idx + 1) % 10 == 0:
#                     print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(data_loader)} - Loss: {losses.item():.4f}", flush=True)

#             except Exception as e:
#                 print(f"Error in batch {batch_idx + 1}: {e}", flush=True)
#                 continue

#         lr_scheduler.step()
#         print(f"[Epoch {epoch + 1}] Completed. Total Loss: {epoch_loss:.4f}", flush=True)

#         torch.save(model.state_dict(), f"retinanet_epoch_{epoch + 1}.pth")
#         print(f"Saved model checkpoint: retinanet_epoch_{epoch + 1}.pth", flush=True)

# if __name__ == "__main__":
#     main()

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights, RetinaNetClassificationHead
from PIL import Image
import os
import json
from tqdm import tqdm
import torchvision.transforms as T
import time
# from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True  # Enable fastest conv algorithms for fixed input sizes


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.image_info = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            self.annotations.setdefault(img_id, []).append(ann)

        self.ids = [img_id for img_id in self.image_info.keys()
                    if img_id in self.annotations and len(self.annotations[img_id]) > 0]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.annotations.get(img_id, []):
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        if len(boxes) == 0:
            # Skip samples with no valid boxes
            return None

        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid box in image ID {img_id}: {box}")

        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


def get_transform():
    def transform(image, target):
        image = T.ToTensor()(image)
        return image, target
    return transform


def collate_fn(batch):
    # Remove Nones
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))



def main():
    dataset_root = "Food-Recognition-1/train/images"
    annotation_file = "Food-Recognition-1/annotations/instances_train.json"
    num_classes = 498  # 497 classes + background

    dataset = CocoDataset(dataset_root, annotation_file, transforms=get_transform())

    # Train on a small subset first for speed
    # dataset = Subset(dataset, range(200))  # ⬅️ change this number when scaling up
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=8,  # Try increasing if GPU memory allows
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=4,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
    # dataset = Subset(dataset, range(10))
    dataset = Subset(dataset, range(100))  # try 500 first, then 1000, etc.
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=2,
    #     pin_memory=False,
    #     persistent_workers=False
    # )
    data_loader = DataLoader(
    dataset,
        batch_size=4,  # Adjust based on GPU memory
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Increase if CPU allows
        pin_memory=True,
        persistent_workers=True
    )

    torch.backends.cudnn.benchmark = False
    # device = torch.device('cpu')  # use CPU for this test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)

    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    scaler = torch.amp.GradScaler("cuda") #GradScaler()
    torch.backends.cudnn.benchmark = True
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        print(f"\n[Epoch {epoch + 1}/{num_epochs}] Starting...", flush=True)
        

        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if any(len(t["boxes"]) == 0 for t in targets):
                    print(f"Skipping batch {batch_idx+1} due to empty target")
                    continue

                # loss_dict = model(images, targets)
                # losses = sum(loss for loss in loss_dict.values())
                # if not torch.isfinite(losses):
                #     print(f"Skipping batch {batch_idx+1} due to non-finite loss: {losses.item()}")
                #     continue

                # optimizer.zero_grad()
                # losses.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimizer.step()
                with torch.amp.autocast("cuda"): #torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    if not torch.isfinite(losses):
                        print(f"Non-finite loss at batch {batch_idx + 1}: {losses.item()}")
                        continue

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()


                epoch_loss += losses.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx + 1}/{len(data_loader)} - Loss: {losses.item():.4f}", flush=True)

            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {e}", flush=True)
                continue

        lr_scheduler.step()
        print(f"[Epoch {epoch + 1}] Completed. Total Loss: {epoch_loss:.4f}", flush=True)
        end_time = time.time()
        print(f"Epoch {epoch+1} took {(end_time - start_time)/60:.2f} minutes")
        torch.save(model.state_dict(), f"retinanet_epoch_{epoch + 1}.pth")
        print(f"Saved model checkpoint: retinanet_epoch_{epoch + 1}.pth", flush=True)


if __name__ == "__main__":
    main()
