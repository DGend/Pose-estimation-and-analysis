from __future__ import print_function
import numpy as np
import cv2
import mediapipe as mp      # pip install mediapipe
import numpy as np
import argparse

# We are declaring the drawing_utils and pose in shortened variables
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# Here we define the mediapipe process, in this case, mediapipe_pose
def process_image(image):
    if image is not None:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            return (image)
    else:
        return None


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    if label == "person":
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # We crop the section of one single person
        src2 = img[y:y_plus_h, x:x_plus_w]
        # Then we process it with mediapipe

        try:
            img2 = process_image(src2)
            img[y:y_plus_h, x:x_plus_w] = img2
        except:
            print("draw_prediction error!\n")

class_path = "./yolo/yolov3.txt"

# # Yolov3
# weights_path = "./yolo/yolov3.weights"
# config_path = "./yolo/yolov3.cfg"

# # Yolov3-tiny
# weights_path = "./yolo/yolov3-tiny.weights"
# config_path = "./yolo/yolov3-tiny.cfg"

# Yolov4-tiny
weights_path = "./yolo/yolov4-tiny.weights"
config_path = "./yolo/yolov4-tiny.cfg"

# For webcam input:
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()

    Width = img.shape[1]
    Height = img.shape[0]
    scale = 0.00392

    classes = None

    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_path, config_path)

    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()

