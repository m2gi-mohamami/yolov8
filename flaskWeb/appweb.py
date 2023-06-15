import json

import numpy as np
from flask import Flask, render_template, Response,jsonify,request,session
import pytesseract
import torch
import os

from flask import Response
from ultralytics import YOLO
import cv2
import math
import json
# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection

app = Flask(__name__)
@app.route('/ocr')
def ocr():
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    cap = cv2.VideoCapture(0)
    # Load the Tesseract OCR model
    print(22222)
    while True:
       c,img=cap.read()
       cv2.imshow("Test Detection",img)
       # Extract text from the image
       text = pytesseract.image_to_string(img)
       # Print the text
       print(text)


       if cv2.waitKey(1) & 0xFF == ord('1'):
           break
    cap.release()
    cv2.destroyAllWindow()

@app.route('/')
def jeson():
    path_x = 0
    video_capture = path_x
    # Create a Webcam Object
    cap =cv2.VideoCapture("http://192.168.137.146:81/stream")
    object = {}
    model=YOLO("../YOLO-Weights/yolov8n.pt")
    # Load the model

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    while True:
        success, img = cap.read()
        results = model(img, stream=True,show=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                if class_name  in object.keys():
                    object[class_name] += 1
                else:
                    object[class_name] = 1

                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            print("object:",object)
            response = Response(json.dumps(object), mimetype="application/json")
            object = {}
            return(response)

app = Flask(__name__)
@app.route('/ocrr')
def ocrr():
    # WebCam
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    cap = cv2.VideoCapture("http://192.168.137.146:81/stream")
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("cannot open video")

    cntr = 0;

    while True:
        ret, frame = cap.read()
        cntr = cntr + 1;
        if ((cntr % 20) == 0):
            imgH, imgW, _ = frame.shape

            x1, y1, w1, h1 = 0, 0, imgH, imgW

            imgchar = pytesseract.image_to_string(frame)
            imgboxes = pytesseract.image_to_boxes(frame)
            for boxes in imgboxes.splitlines():
                boxes = boxes.split(' ')
                x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                cv2.rectangle(frame, (x, imgH - y), (w, imgH - h), (0, 0, 255), 3)
            cv2.putText(frame, imgchar, (x1 + int(w1 / 50), y1 + int(h1 / 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(imgchar)

            cv2.imshow('Text Dtection Tutorial', frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/home')
def home():
    from multiprocessing import Process
    import gevent
    def model1():
      # Do something with the data using model 1.
       jeson()

    def model2():
        ocrr()

    t1 = gevent.spawn(model1)
    t2 = gevent.spawn(model2)

    # Run the tasks in parallel
    gevent.joinall([t1, t2])


if __name__ == "__main__":
    app.run(debug=True)