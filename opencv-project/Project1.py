import math
import time

import cv2
import mediapipe as mp
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandDetectionModule as HDM

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange= volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar =400
volPer = 0
wCam,hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
cTime = 0
pTime = 0
detector = HDM.handDetector(detectionCon=0.7)
while True:
  success,img = cap.read()
  img = detector.findHands(img)
  lmList = detector.findLandMark(img,draw=False)
  if len(lmList)!=0:
    x1,y1 = lmList[4][1],lmList[4][2]
    x2,y2 = lmList[8][1],lmList[8][2]
    cx,cy = (x1+x2)//2,(y1+y2)//2

    cv2.circle(img,(x1,y1),10,(255,0,255),-1)
    cv2.circle(img,(x2,y2),10,(255,0,255),-1)
    cv2.circle(img,(cx,cy),10,(255,0,255),-1)
    cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
    length = math.hypot(x2-x1,y2-y1)
    vol = np.interp(length,[50,300],[minVol,maxVol])
    volBar = np.interp(length,[50,300],[400,150])
    volPer = np.interp(length,[50,300],[0,100])
    volume.SetMasterVolumeLevel(vol, None)
    cv2.putText(img, f'Volume:{int(volPer)}%', (50, 450),
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
    if(length<50):
       cv2.circle(img,(cx,cy),10,(0,255,0),-1)
  
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
