import math
import time

import cv2
import mediapipe as mp


class poseDetector():
  def __init__(self,mode =False,upBody =False,smooth = True,detectionCon = 0.5,trackCon =  0.5):
    self.mode = mode
    self.upBody = upBody
    self.smooth = smooth
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose( static_image_mode= self.mode,smooth_landmarks=self.smooth,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
    # self.pose = self.mpPose.Pose( self.mode,self.smooth,self.detectionCon,self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils
  
  def findPose(self,img,draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.result = self.pose.process(imgRGB)
    if self.result.pose_landmarks:
      if draw:
        self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,
                              self.mpPose.POSE_CONNECTIONS)
    return img
  def findPosition(self,img,handNo=0,draw = True):
      self.lmList = []
      if self.result.pose_landmarks:
        for id, lm in enumerate(self.result.pose_landmarks.landmark):
          h, w, c = img.shape
          cx, cy = int(lm.x*w), int(lm.y * h)
          self.lmList.append([id,cx,cy])
          if draw:
            cv2.circle(img, (cx, cy), 7, (255, 0, 255), -1)
      return self.lmList
  
  def findAngle(self,img,p1,p2,p3,draw =True):
    x1,y1 = self.lmList[p1][1:]
    x2,y2 = self.lmList[p2][1:]
    x3,y3 = self.lmList[p3][1:]
    angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y2-y2,x2-x1))
    if draw:
      cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
      cv2.line(img,(x2,y2),(x3,y3),(255,255,255),3)
      cv2.circle(img,(x1,y1),10,(0,0,255),-1)
      cv2.circle(img,(x2,y2),10,(0,0,255),-1)
      cv2.circle(img,(x3,y3),10,(0,0,255),-1)
      cv2.circle(img,(x1,y1),15,(0,0,255),2)
      cv2.circle(img,(x2,y2),15,(0,0,255),2)
      cv2.circle(img,(x3,y3),15,(0,0,255),2)
      cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)










def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    poseDetection  = poseDetector()
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


if __name__ == "__main__":
    main()
