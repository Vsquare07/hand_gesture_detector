from handtrackermodule import handDetector
import cv2
import time
import torch

model = torch.load(f="models/model.pth", weights_only=False)

cap = cv2.VideoCapture(0)
hands = handDetector(maxHands=1)

pTime = 0
cTime = 0
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break

    img = cv2.resize(img, (640, 400))
    img = hands.findHands(img)
    landmarks = hands.findPosition(img)

    #TODO implement preds with softmax and make a bounding box for hand
    preds = model()

    cTime = time.time()
    fps = int(1//(cTime-pTime))
    pTime = cTime
    cv2.putText(img, "FPS = "+str(fps), (30,60), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0), 2)
    cv2.imshow("Hand Landmarks", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
