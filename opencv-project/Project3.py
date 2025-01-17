import time

import cv2
import mediapipe as mp
import numpy as np

import PoseDetectionModule

wCam,hCam = 1280 ,720
cTime =0
pTime = 0
poseDetection = PoseDetectionModule.poseDetector()
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
while True:
  success , img = cap.read()
  img = poseDetection.findPose(img)
  lmList = poseDetection.findPosition(img,draw=False)
  if len(lmList)!=0: 
    poseDetection.findAngle(img,12,14,16)
    poseDetection.findAngle(img,11,13,15)
  # print(lmList)
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
      cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
