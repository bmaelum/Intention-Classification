#!/usr/bin/env python

#import rospy

import cv2
import numpy as np
import sys
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

#cap = cv2.VideoCapture("/home/project/Bjorn/SonyHandycamTest.mp4")
cap = cv2.VideoCapture(-1)



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/media/project/MEDIAPARTIT/dataCollection/FER' + sys.argv[1] + '.avi',fourcc, 20.0, (640,480))

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)
    out.write(frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
