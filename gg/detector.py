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

video = UMatFileVideoStream('Gol.mp4', 30).start()
rgb = cv2.UMat(self.height, self.width, cv2.CV_8UC3)

while not video.stopped:
    cv2.cvtColor(video.read(), cv2.COLOR_BGR2RGB, hsv, 0)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, hsv, 0)
    img = hsv.get()
    cv2.cvtColor(frame, cv2.COLOR_RGB2HSV, hsv, 0)
    try:
        img = cv2.resize(img, None, fx=0.4, fy=0.4)

    except Exception as e:
        print(str(e))

    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
     swapRB=True, crop=False)

scale = 0.5

gpu_frame = cv2.cuda_GpuMat()

while ret:

    gpu_frame.upload(frame)

    resized = cv2.cuda.resize(gpu_frame, (int(1280 * scale), int(720 * scale)))

    luv = cv2.cuda.cvtColor(resized, cv.COLOR_BGR2LUV)
    hsv = cv2.cuda.cvtColor(resized, cv.COLOR_BGR2HSV)
    gray = cv2.cuda.cvtColor(resized, cv.COLOR_BGR2GRAY)
    
    # download new image(s) from GPU to CPU (cv2.cuda_GpuMat -> numpy.ndarray)
    resized = resized.download()
    luv = luv.download()
    hsv = hsv.download()
    gray = gray.download()

    ret, frame = vod.read()

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

    cv2.imshow("Image",cv2.resize(img, (800,600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


















print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()