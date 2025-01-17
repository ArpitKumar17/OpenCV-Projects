import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0
mpFaces = mp.solutions.face_mesh
faces = mpFaces.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
mpDrawingSt = mp.solutions.drawing_styles
while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faces.process(imgRGB)
    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
          mpDraw.draw_landmarks(img, facelms, mpFaces.FACEMESH_CONTOURS,connection_drawing_spec= mpDrawingSt.get_default_face_mesh_contours_style())
    cv2.imshow("Image", img)
    cv2.waitKey(1)
