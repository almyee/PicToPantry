# -------------------------------------- load Fast-RNN and Retina-Net models----------------------------------
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# # Load Faster R-CNN pretrained on COCO
# # model = fasterrcnn_resnet50_fpn(pretrained=True)
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
# model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# model.eval()


# from torchvision.models.detection import retinanet_resnet50_fpn

# # Load RetinaNet pretrained on COCO
# # model = retinanet_resnet50_fpn(pretrained=True)
# from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
# model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
# model.eval()

import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Load pre-trained object detection models
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
retinanet = models.detection.retinanet_resnet50_fpn(weights="DEFAULT").eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
faster_rcnn.to(device)
retinanet.to(device)

# Image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Paths
input_dir = './images'
output_dir_frcnn = './outputs/faster_rcnn'
output_dir_retina = './outputs/retinanet'
os.makedirs(output_dir_frcnn, exist_ok=True)
os.makedirs(output_dir_retina, exist_ok=True)

# Optional: load real COCO label names (simplified version)
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush"
}

def draw_and_save(image_path, outputs, out_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > threshold:
            box = box.tolist()
            label = label.item()
            score = score.item()
            draw.rectangle(box, outline='red', width=2)
            text = f"{COCO_LABELS.get(label, str(label))} ({score:.2f})"
            # text = f"{COCO_LABELS.get(label, str(label))} ({score:.2f})"
            draw.text((box[0], box[1] - 10), text, fill='red', font=font)

    image.save(out_path)

# Run inference
for img_file in Path(input_dir).glob("*.[jp][pn]g"):
    image = Image.open(img_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Faster R-CNN
    with torch.no_grad():
        preds_frcnn = faster_rcnn(img_tensor)[0]
    draw_and_save(img_file, preds_frcnn, f"{output_dir_frcnn}/{img_file.stem}_frcnn.jpg")

    # RetinaNet
    with torch.no_grad():
        preds_retina = retinanet(img_tensor)[0]
    draw_and_save(img_file, preds_retina, f"{output_dir_retina}/{img_file.stem}_retina.jpg")

print("✅ Inference complete. Results saved to ./outputs/")

