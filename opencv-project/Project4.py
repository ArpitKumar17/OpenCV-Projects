import os
import time

import cv2
import mediapipe as mp
import numpy as np

import HandDetectionModule as HDM

eraserThickness =35
brushThickness =15
xp,yp=0,0
wCam,hCam = 1050,720
folderName = 'Photos'
myList=  os.listdir(folderName)
print(myList)
overlayImg = []
for imPath in myList:
  headerImg = cv2.imread(f'{folderName}/{imPath}')
  overlayImg.append(headerImg)
print(len(overlayImg))
header = overlayImg[0]
tipIds = [4,8,12,16,20]
cTime=0
pTime = 0
pointer = HDM.handDetector(detectionCon=0.75)
cap = cv2.VideoCapture(0)
drawColor = (255,0,255)
cap.set(3,wCam)
cap.set(4,hCam)
imgCanvas =  np.zeros((540,960,3),np.uint8)
while True:
  success , img =cap.read()
  img = cv2.flip(img,1)
  img = pointer.findHands(img)
  lmList = pointer.findLandMark(img,draw= False)
  if len(lmList)!=0:
    x1,y1 = lmList[8][1:]
    x2,y2 = lmList[12][1:]
    raisedUp = pointer.fingerUp()
    if raisedUp[1] and raisedUp[2]:
      xp,yp=0,0
      if y1<103:
        if 100<x1<250:
          drawColor = (255,0,255)
          header = overlayImg[0]
        elif 350<x1<450:
          drawColor = (255,0,0)
          header = overlayImg[1]
        elif 550 <x1<650:
          drawColor = (0,255,0)
          header = overlayImg[2]
        elif 850 < x1<1000:
          drawColor = (0,0,0) 
          header = overlayImg[3]
      cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
    if raisedUp[1] and raisedUp[2]==False:
      cv2.circle(img,(x1,y1),15,drawColor,-1)
      if(xp == 0 and yp == 0):
        xp,yp =x1,y1
      if drawColor == (0,0,0):
        cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
        cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
      cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
      cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
      xp,yp =x1,y1
  
  imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
  _,imgInv =  cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
  imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
  img = cv2.bitwise_and(img,imgInv)
  img = cv2.bitwise_or(img,imgCanvas)
  header_resized = cv2.resize(header, (960, 103))
  img[0:103,0:960]=header_resized
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
      cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
  
  cv2.imshow("Image",img)
  #cv2.imshow("Image Canvas",imgCanvas)
  cv2.waitKey(1)
