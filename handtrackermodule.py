import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    landmark_spec = self.mpDraw.DrawingSpec(color=(119, 52, 235), thickness=2, circle_radius=4)  # Red points
                    connection_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green lines
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS, 
                                               landmark_drawing_spec = landmark_spec, 
                                               connection_drawing_spec=connection_spec)
        return img
    def findPosition(self, img, tracking_id = None, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            if tracking_id is None:
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            else:
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == tracking_id:
                        x = cx
                        y = cy
                        if draw:
                            cv2.circle(img, (x, y), 15, (255, 0, 255), cv2.FILLED)
                    lmList.append([id, cx, cy])
        return lmList