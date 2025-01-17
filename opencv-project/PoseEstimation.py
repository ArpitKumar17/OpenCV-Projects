import time

import cv2
import mediapipe as mp

import PoseDetectionModule as pm

cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
poseDetection  = pm.poseDetector()
while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    img = poseDetection.findPose(img)
    lmList = poseDetection.findPosition(img)
    print(lmList)
    cv2.putText(img, str(int(fps)), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
