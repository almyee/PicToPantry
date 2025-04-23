# https://inference.roboflow.com/quickstart/explore_models/
# from roboflow import Roboflow # run the 4 lines below to download fridge-objects data
# rf = Roboflow(api_key="1D3URBqbnuzbjSo5V6uK")
# project = rf.workspace("fooddetection-essdj").project("fridge-objects")
# dataset = project.version(1).download("yolov8")
# -----------------------------------
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load pretrained YOLOv8 model (you can use 'yolov8n', 'yolov8s', etc.)
model = YOLO("yolov8n.pt")

# Train your model on the fridge dataset (optional)
# model.train(data="Fridge-Objects-1/data.yaml", epochs=20)

# Run inference
results = model("images/image3.png")

# Show results with bounding boxes
results[0].show()
# annotated_img = results[0].plot()
# plt.imshow(annotated_img)
# plt.axis('off')
# plt.show()

# -------------------------------------
# from io import BytesIO
# import requests
# import supervision as sv
# from inference import get_model
# from PIL import Image
# from PIL.ImageFile import ImageFile
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key= "1D3URBqbnuzbjSo5V6uK" #"rf_qTIJ3SlO2gUdyKcz2xDBtABRbAx1"  "API_KEY"
# )

# result = CLIENT.infer("images/image.png", model_id="fridge-objects/1")
# print(result)

# ----------------------------------------------
# def load_image_from_url(url: str) -> ImageFile:
#     response = requests.get(url)
#     response.raise_for_status()  # check if the request was successful
#     image = Image.open(BytesIO(response.content))
#     return image


# # load the image from an url
# image = load_image_from_url("https://media.roboflow.com/inference/people-walking.jpg")
# image = "images/image.png"
# # # load a pre-trained yolov8n model
# # model = get_model(model_id="yolov8n-640")
# # # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
# # results = model.infer(image)[0]
# # # load the results into the supervision Detections api
# detections = sv.Detections.from_inference(result)

# # create supervision annotators
# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # annotate the image with our inference results
# annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# # display the image
# sv.plot_image(annotated_image)
# --------------------------
# from inference import get_model
# import supervision as sv
# import cv2

# # define the image url to use for inference
# image_file = "images/image.png" #"taylor-swift-album-1989.jpeg"
# image = cv2.imread(image_file)

# # load a pre-trained yolov8n model
# model = get_model(model_id="fridge-objects/1") 

# # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
# results = model.infer(image)[0]

# # load the results into the supervision Detections api
# detections = sv.Detections.from_inference(results)

# # create supervision annotators
# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # annotate the image with our inference results
# annotated_image = bounding_box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections)

# # display the image
# sv.plot_image(annotated_image)
