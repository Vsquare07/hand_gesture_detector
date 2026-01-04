from handtrackermodule import handDetector
import cv2
import time
import torch
import torch.nn as nn
import numpy as np

model = torch.load(f="models/model1.pth", weights_only=False)
model = model.to(device = "cpu")

cap = cv2.VideoCapture(0)
hands = handDetector(maxHands=1)
classes = ["dislike", "like", "none"]

pTime = 0
cTime = 0
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break

    img = cv2.resize(img, (640, 400))
    img_np = np.array(img)
    showHands = hands.findHands(img_np)
    landmarks = hands.findPosition(img_np)
    if(len(landmarks)!=0):
        lms = []
        for x,y in landmarks:
            x = float(x)
            y = float(y)
            x/=640
            y/=400
            lms.append(x)
            lms.append(y)

        with torch.inference_mode():
            pred = model(torch.tensor(lms, dtype=torch.float))
            pred_prob = torch.softmax(pred, dim=0)
            pred_idx = torch.argmax(pred_prob, dim=0).item()
            cv2.putText(img, classes[pred_idx], landmarks[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),1)
            str_probs = "dislike = "+str(int(pred[0].item()*100))+" like = "+str(int(pred[1].item()*100))+" none = "+str(int(pred[2].item()*100))
            cv2.putText(img, str_probs, (200,50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0), 1)
    cTime = time.time()
    fps = int(1//(cTime-pTime))
    pTime = cTime
    cv2.putText(img, "FPS = "+str(fps), (30,60), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
    cv2.imshow("Hand Landmarks", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
