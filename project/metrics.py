# this file pulls the metrics from YOLO trained batches
from ultralytics import YOLO
import json
import os

# Load model
model = YOLO('./runs/detect/train5/weights/best.pt')

# Run validation
metrics = model.val(data='/root/ecs271_files/PicToPantry/project/Food-Recognition-1/data.yaml', project="./runs/detect") # need to put full static path for this to work

# print(metrics.box.map)         # mAP@0.5
# print(metrics.box.map50)       # mAP@0.5
# print(metrics.box.map75)       # mAP@0.75
# print(metrics.box.maps)        # list of mAP per class
# print(metrics.box.precision)   # overall precision
# print(metrics.box.recall)      # overall recall

# Extract results
results = {
    "mAP_0.5": metrics.box.map50,            # no parentheses
    "mAP_0.5:0.95": metrics.box.map,
    "mAP_0.75": metrics.box.map75,
    "mAP_per_class": metrics.box.maps.tolist(),
    "mean_precision": metrics.box.mp,        # <- fixed: no parentheses
    "mean_recall": metrics.box.mr            # <- fixed: no parentheses
}

# Define output path
output_path = os.path.join(
    "C:/Users/mdhar/OneDrive/Documents/ECS_271/PicToPantry/project/runs/detect/val7",
    "metrics_results.json"
)

# Save results to a JSON file
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Metrics saved to {output_path}")


'''
Results saved to ./runs/detect/val6

yolo predict model=your_model.pt source=valid/images save_conf save_txt

'''