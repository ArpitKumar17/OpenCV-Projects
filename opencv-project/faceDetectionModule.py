import time

import cv2
import mediapipe as mp


class faceDetection:
  def __init__(self,mode=False,detectionCon=0.5):
    self.mode = mode
    self.detectionCon = detectionCon
    self.mpFace = mp.solutions.face_detection
    self.faces = self.mpFace.FaceDetection()
    self.mpDraw = mp.solutions.drawing_utils
  def findFace(self,img,draw =True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.result = self.faces.process(imgRGB)
    if self.result.detections:
      if draw:
        for detection in self.result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin*iw) , int(bboxC.ymin*ih),\
              int(bboxC.width*iw),int(bboxC.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

    return img










def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    faceDetector = faceDetection()
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        img = faceDetector.findFace(img)

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
