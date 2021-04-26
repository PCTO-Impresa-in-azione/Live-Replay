import cv2
import numpy as np

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("cfg/yolov3.weights", "cfg/yoloNetwork.cfg")
#save all the names in file o the list classes
classes = []
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

#rilevazione
video_capture = cv2.VideoCapture("resources/prova2.mp4")
width= int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer= cv2.VideoWriter('risultati/risultato.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))

while True:
    # Capture frame-by-frame
    re,img = video_capture.read()
    try:
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    except:
        break

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

    writer.write(img)

    cv2.imshow("Image",cv2.resize(img, (400,400)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
writer.release()
cv2.destroyAllWindows()