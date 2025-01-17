import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui as p

import HandDetectionModule as HDM

###################
wCam,hCam =640,480
wScr,hScr =p.size().width,p.size().height
frameR =100
smoothEff = 1.5
###################
pLocX,pLocY = 0,0
cLocX,cLocY = 0,0
cTime =0 
pTime = 0 
handDetect = HDM.handDetector(detectionCon=0.75)
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
xm=0
ym  = 0
while True: 
  success,img = cap.read()
  img = cv2.flip(img,1)
  img = handDetect.findHands(img)
  lmList = handDetect.findLandMark(img, draw=False)
  upFingers = handDetect.fingerUp()
  if len(lmList)!=0:
    x1,y1 = lmList[8][1:]
    x2,y2 = lmList[12][1:]
    cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),3)
    if upFingers[1] and upFingers [2]:
      length = math.hypot(x2-x1,y2-y1)
      cx,cy = (x1+x2)//2,(y1+y2)//2
      cv2.circle(img,(x1,y1),10,(255,0,255),-1)
      cv2.circle(img,(x2,y2),10,(255,0,255),-1)
      cv2.circle(img,(cx,cy),10,(255,0,255),-1)
      cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
      if length<40:
        cv2.circle(img,(cx,cy),10,(0,255,0),-1)
        p.click()

    if upFingers[1] and upFingers[2] == False:
      xm =np.interp(x1,(frameR,wCam-frameR),(0,wScr))
      ym =np.interp(y1,(frameR,hCam-frameR),(0,hScr))
      cLocX = pLocX + (xm-pLocX)/smoothEff
      cLocY = pLocY + (ym-pLocY)/smoothEff
      p.moveTo(cLocX, cLocY)
      cv2.circle(img,(x1,y1),10,(0,255,0),-1)
      pLocX,pLocY = cLocX,cLocY
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
      cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
