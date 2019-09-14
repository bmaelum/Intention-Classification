#!/usr/bin/env python

import cv2
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

from expressionRecognition import detectFace, cropFaces, displayCrop, predictImg, detectAndCropColor,facialExpressionRecogntion
from headMovementEstimation import mostCommon, get_face, calculateDirection

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib

print sys.argv[1]
videoName = sys.argv[1]

camera = 'webcam'

CNN_INPUT_SIZE = 128

def intentClass(whichIntent, faceExpModel, faceCascade, detector, predictor):

    predictExp = 0
    predictHeadMov = 0
    predictExp = 0
    goFER = 0
    goHPC = 0
    #if whichIntent == 0:
        # Do nothing
    if whichIntent == 1:
        predictExp = 1
        goFER = 1
    elif whichIntent == 2:
        predictHeadMov = 1
        goHPC = 1
    elif whichIntent == 3:
        predictExp = 1
        predictHeadMov = 1
        goFER = 1
        goHPC = 1

    instructions = 0
    print "--- Intention classification for communication between a human and a robot ---"
    if instructions == 1:
        print "First you will be required to present a facial expression before you will do a head movement."
        print "If done correctly these gestures will be detected by the robot and it will perform the desired task."
        raw_input("Press Enter to continue...")

    if instructions == 1:
        print "This is the module for facial expression recognition."
        print "This program can detect the emotions: Happy and Angry."
        print "The program will look for the expression for 3 seconds."

    noseMarks = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    HPCcounter = 0
    FERcounter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    numXPoints = 0
    numYPoints = 0
    sumX = 0
    sumY = 0

    currentTime = 0
    previousTime1 = 0
    previousTime2 = 0

    directionArray = []
    moveSequence = []
    moves = []
    classifyMoves = 0
    headPoseDirection = 'emtpy'


    if camera == 'webcam':
        #video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('/media/project/MediaPartition/ExperimentData/Webcam/'+ videoName +'.avi',fourcc, 20.0, (640,480))

    ## Facial Expression Recognition variables
    FER_prediction = []
    FERclass = ''
    FERstart = 0
    classifyMoves = 0

    ## Head movement variables
    counter = 0
    HeadMov = []
    HMCclass = ''
    detectedFaces = []
    mark = []

    notDone = 0
    progressStatus = [0,0]

    while notDone in progressStatus: # This waits for each module to finsih

        if camera == 'webcam':
            # Capture frame-by-frame
            ret, color = video_capture.read()
            color = cv2.flip(color, 1)

        if predictExp == 1:
            if FERcounter == 0:
                FERstart = time.time()
                FERcounter += 1
            currentTime = time.time()
            if currentTime - FERstart < 3:
                FER_prediction.append(facialExpressionRecogntion(color, faceExpModel, faceCascade))
            else:
                predictExp = 2
                FER_prediction = filter(None, FER_prediction)
                FERclass = mostCommon(FER_prediction)
                FERclass = FERclass[2:7]
                predictHeadMov = 0
                if FERclass == '':
                    print("Did not detect an expression, try again.")
                    predictExp = 1
                    FERcounter = 0
                    FER_prediction = []
                else:
                    progressStatus[0] = 1
                    if whichIntent == 1:
                        progressStatus[1] = 1
                    print "Detected expression: " + str(FERclass)
                    print "Progress: FER DONE"

        if predictHeadMov == 1 and predictExp != 1:
            if HPCcounter == 0:
                timer = time.time()
                previousTime1 = timer
                previousTime2 = timer
                HPCcounter += 1
                #raw_input()

            moveTime = 2 # number of seconds for head movement classification
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            shape = []
        	# loop over the face detections
            for rect in rects:
        		# determine the facial landmarks for the face region, then
        		# convert the facial landmark (x, y)-coordinates to a NumPy
        		# array
        		shape = predictor(gray, rect)
        		shape = face_utils.shape_to_np(shape)

            if len(shape) > 0:
                #print len(shape)
                # # Get average position of nose
                noseMarksTemp = []
                noseMarksTemp.append(shape[30][0])
                noseMarksTemp.append(shape[30][1])
                noseMarks[0] = noseMarksTemp

                for i in range(9, 0, -1):
                    noseMarks[i] = noseMarks[i-1]

                # Get the direction of head movement
                headPoseDirection = calculateDirection(noseMarks)



                directionArray.append(headPoseDirection)
            if headPoseDirection is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color, headPoseDirection, (20,70), font, 1, (0,255,0), 4)

            if classifyMoves == 0:
                classifyMoves = 1
                timer = time.time()
                previousTime1 = timer
                previousTime2 = timer

            currentTime = time.time()
            if currentTime - previousTime1 > moveTime and classifyMoves == 1:
                print "------------------------------------------------"
                print "Elapsed timer 1: " + str(currentTime - previousTime1)

                # Get most common direction
                if len(directionArray) > 0:
                    HMCclass = mostCommon(directionArray)
                    previousTime1 = currentTime
                    classifyMoves = 2

                    progressStatus[1] = 1
                    predictHeadMov = 0
                    if whichIntent == 2 and len(directionArray):
                        progressStatus[0] = 1
                        print "Progress: HMC DONE"
                else:
                    predictHeadMov = 1
                    HPCcounter = 0
                    classifyMoves = 0
                    raw_input("No head movements detected, try again...")

                directionArray = []

        elif predictHeadMov == 2:
            progressStatus[1] = 1
            print "Skipped Head Movement Estimation."
            break

        cv2.imshow("", color)
        out.write(color)
        if camera == 'kinect':
            listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    if camera == 'webcam' or camera == 'video':
        video_capture.release()
        out.release()

    cv2.destroyAllWindows()

    intentionClassification = [FERclass, HMCclass]
    if goFER and goHPC:
        return FERclass, HMCclass
    elif goFER and not goHPC:
        return FERclass
    elif goHPC and not goFER:
        return HMCclass

def fromTerminal():
    # Load Facial Expression Recognition trained model
    print "- Loading FER model... -"
    #faceExpModel = tf.keras.models.load_model("/home/bjornar/ML_models/FER/Good models(80+)/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")
    faceExpModel = tf.keras.models.load_model("/home/project/Bjorn/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")

    # Load Face Cascade for Face Detection
    print "- Loading Face Cascade for Face Detection... -"
    cascPath = "/home/project/IntentionClassification-Repository/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/project/my_jaco_package/src/bjornPkg/scripts/shape_predictor_68_face_landmarks.dat")

    intentClass(1, faceExpModel, faceCascade, detector, predictor)

if sys.argv[1] == 'terminal':
    fromTerminal()
