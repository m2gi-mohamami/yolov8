import cv2
from ultralytics import YOLO
model=YOLO('yolov8n.pt')
results=model('../Imgs/3.jpeg',show=True)

cv2.waitKey(0)