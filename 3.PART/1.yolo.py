"""

YOLO (You Only Look Once) is a single-stage object detection method that uses a CNN to detect objects in an image.

How YOLO works:
1. Grid Division: Divides the image into a grid of cells.
2. Bounding Box Prediction: Each cell predicts bounding boxes and confidence scores.
3. Class Prediction: Each cell predicts class probabilities.
4. Non-Maximum Suppression: Filters overlapping boxes to keep the best ones.

How To use YOLO:
1. Load the pre-trained YOLO model.
2. Preprocess the image (resize, normalize).
3. Pass the image through the model.
4. Postprocess the output (filter, draw bounding boxes).

code example:
from ultralytics import YOLO

# 1. Load the pre-trained YOLO model
model = YOLO("yolov8n.pt")

# 2 & 3. Preprocess and pass the image through the model
results = model("path/to/image.jpg")

# 4. Postprocess the output (display results)
for result in results:
    result.show()
    result.save(filename="result.jpg")


"""