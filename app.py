import cv2
import sys
import torch
from yolov5_face import detect_face
from keras.models import model_from_json
import numpy as np

# Camera
source = 0
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

# face emotion predictor
with open('./weight/fer.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./weight/fer.h5")

# running
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cam = cv2.VideoCapture(source)
while cam.isOpened():
    ret, frame = cam.read()
    frame = cv2.resize(frame, new_frame)

    # flip the frame
    if flip:
        frame = cv2.flip(frame, 1)

    if ret:
        # detect face
        _, _, boxes, conf = model.detect_one(frame, new_size=256, conf_thres=0.5)

        if len(boxes):
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]

                # draw a rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=1, lineType=cv2.LINE_AA)

                # crop/ getting features/ predict
                cropped_img = frame[y1:y2, x1:x2, :]
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
                resized_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
                cv2.normalize(resized_img, resized_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

                print(resized_img.shape)
                probs = loaded_model.predict(resized_img)[0]
                yhat = int(np.argmax(probs))
                emotion = labels[yhat]
                prob = probs[yhat]

                if prob > 0.8:
                    cv2.putText(frame, emotion, (x1, y1 - 10), font_face, font_scale, box_color, 2)
                    cv2.putText(frame, f"Prob: {prob:.3f}", (x1, y2 + 30), font_face, font_scale, font_color, 2)
                else:
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), font_face, font_scale, warning_color, 1)

    cv2.imshow('Face', frame)
    if cv2.waitKey(1) == ord('e'):
        break

cam.release()
cv2.destroyAllWindows()
