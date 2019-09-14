import cv2
import numpy as np

video_capture = cv2.VideoCapture(-1)
print video_capture.isOpened()
ret, color = video_capture.read()
cv2.imwrite("frame.png", color)

while True:
    ret, color = video_capture.read()

    cv2.imshow("frame", color)

video_capture.release()
cv2.destroyAllWindows()
