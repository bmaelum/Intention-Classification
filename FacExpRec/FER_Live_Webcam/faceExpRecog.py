import cv2
import os
from FacExpLib import detectFace

# Create new haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read from camera
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    gray, frame = detectFace(frame, faceCascade)

    cv2.imshow("gray", gray)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.waitKey(0)

#cv2.VideoCapture.release()
