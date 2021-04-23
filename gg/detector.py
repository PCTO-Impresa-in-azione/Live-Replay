from __future__ import division

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from UMatFileVideoStream import UMatFileVideoStream

CUDA = torch.cuda.is_available()


net = cv2.dnn.readNet("cfg/yolov3.weights", "cfg/yoloNetwork.cfg")
num_classes = 80
classes = []
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vcap = cv2.VideoCapture('Gol.mp4')

width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`



video = UMatFileVideoStream('Gol.mp4', 30).start() #creato l'oggetto video di classe UmatFileVideoStream (Umat serve a processarlo tramite la gpu)
rgb=cv2.UMat(height, width)

while not video.stopped:
    cv2.cvtColor(video.read(), cv2.COLOR_BGR2RGB, hsv, 0)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, hsv, 0)
    img = hsv.get()

  # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
     swapRB=True, crop=False)
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 3)

    cv2.imshow("Image",cv2.resize(img, (800,600))) #mostra il frame attuale del video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vcap.release()
cv2.destroyAllWindows()














print("SUMMARY")


torch.cuda.empty_cache()