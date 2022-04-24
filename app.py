import cv2
import sys
import torch
from yolov5_face import detect_face

# Camera
source = 1
new_frame = (960, 540)
flip = True

# Bounding boxes
box_color = (0, 255, 0)

# text
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
warning_color = (0, 0, 255)

# detector
sys.path.insert(0, 'yolov5_face')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = './weight/yolov5n-0.5.pt'
model = detect_face.face_detection(weight, device)

# running
cam = cv2.VideoCapture(source)
while cam.isOpened():
    ret, frame = cam.read()
    frame = cv2.resize(frame, new_frame)

    # flip the frame
    if flip:        frame = cv2.flip(frame, 1)

    if ret:
        # detect face
        _, _, boxes, conf = model.detect_one(frame, new_size=256, conf_thres=0.5)

        if len(boxes):
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]

                # draw a rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('Face', frame)
    if cv2.waitKey(1) == ord('e'):
        break

cam.release()
cv2.destroyAllWindows()
