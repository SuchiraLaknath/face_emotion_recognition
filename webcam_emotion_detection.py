import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2
from deepface import DeepFace

face_classifire = cv2.CascadeClassifier()
face_classifire.load("models/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifire.detectMultiScale(frame_gray)
    response = DeepFace.analyze(frame, actions= ("emotion"), enforce_detection=False)
    # print(response)
    emotion = response[0]["dominant_emotion"]
    for face in faces:
        x,y, w, h = face
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255,0,0), thickness= 2)
        font = cv2.FONT_HERSHEY_SIMPLEX 
  
        # org 
        org = (x, y) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        image = cv2.putText(frame, emotion, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow("output", frame)
    if (cv2.waitKey(30) == 27):
        break
cap.release()
cv2.destroyAllWindows()