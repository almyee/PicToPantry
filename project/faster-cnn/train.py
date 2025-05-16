import torch
from engine import train_one_epoch, evaluate
from datasets.coco_dataset import CocoDataset
import transforms as T
import utils
import torchvision

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = CocoDataset("Food-Recognition-1/train/images", "Food-Recognition-1/annotations/instances_train.json", get_transform(train=True))
    dataset_valid = CocoDataset("Food-Recognition-1/valid/images", "Food-Recognition-1/annotations/instances_valid.json", get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    num_classes = 1 + len(dataset.coco.getCatIds())  # +1 for background
    model = get_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(10):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_valid, device=device)

    torch.save(model.state_dict(), "fasterrcnn_finetuned.pth")

if __name__ == "__main__":
    main()
