import time

import cv2
import mediapipe as mp


class faceMesh():
  def __init__(self,mode = False,detectionCon = 0.5,trackingCon =0.5):
    self.mode = mode
    self.detectionCon = detectionCon
    self.trackingCon  = trackingCon
    self.mpFaces = mp.solutions.face_mesh
    self.faces = self.mpFaces.FaceMesh(max_num_faces=1)
    self.mpDraw = mp.solutions.drawing_utils
    self.drawSpec = self.mpDraw.DrawingSpec(color = (0,255,0),thickness =1,circle_radius =2)

  def findFM(self,img,draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.result = self.faces.process(imgRGB)
    if self.result.multi_face_landmarks:
      if draw:
        for facelms in self.result.multi_face_landmarks:
            self.mpDraw.draw_landmarks(img, facelms, self.mpFaces.FACEMESH_CONTOURS,
                                  connection_drawing_spec=self.drawSpec)
    return img
  def findLandMarks(self, img):
    lmList = []
    if self.result.multi_face_landmarks:
        for face_landmarks in self.result.multi_face_landmarks:
            ih, iw, ic = img.shape
            for id, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * iw), int(lm.y * ih)
                lmList.append((id, x, y)) 
    return lmList





def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detectFM = faceMesh()
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        img = detectFM.findFM(img)
        lmList = detectFM.findLandMarks(img)
        print(lmList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
