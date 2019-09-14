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
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from multiprocessing import Process, Queue

camera = 'webcam'

if camera == 'kinect':
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame
    from pylibfreenect2 import createConsoleLogger, setGlobalLogger
    from pylibfreenect2 import LoggerLevel

CNN_INPUT_SIZE = 128

def main():
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
    raw_input("To proceed to Facial Expression Recognition press Enter...")
    predictExp = 0

    # Load Facial Expression Recognition trained model
    print "- Loading FER model... -"
    #faceExpModel = tf.keras.models.load_model("/home/bjornar/ML_models/FER/Good models(80+)/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")
    faceExpModel = tf.keras.models.load_model("/home/project/Bjorn/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")

    # Load Face Cascade for Face Detection
    print "- Loading Face Cascade for Face Detection... -"
    #cascPath = "/home/bjornar/MScDissertation/TrainingData/FaceDetection/haarcascade_frontalface_default.xml"
    cascPath = "/home/project/Bjorn/IntentionClassification-Repository/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    ## Initializing Head Movement variables
    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    sample_frame = cv2.imread("sample_frame.png")
    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]

    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    noseMarks = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    counter = 0
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

    if camera == 'kinect':
        ## Initialize Kinect camera
        print "Initializing camera..."
        try:
            from pylibfreenect2 import OpenGLPacketPipeline
            pipeline = OpenGLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenCLPacketPipeline
                pipeline = OpenCLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                pipeline = CpuPacketPipeline()
        #print("Packet pipeline:", type(pipeline).__name__)

        # Create and set logger
        #logger = createConsoleLogger(LoggerLevel.Debug)
        setGlobalLogger()

        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        if num_devices == 0:
            print("- No device connected! -")
            sys.exit(1)

        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial, pipeline=pipeline)

        listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)

        # Register listeners
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)

        device.start()

        # NOTE: must be called after device.start()
        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

    elif camera == 'webcam':
        #video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture(-1)

    elif camera == 'video':
        #video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture("/home/project/Bjorn/SonyHandycamTest.mp4")


    ## Facial Expression Recognition variables
    FER_prediction = []
    FERclass = ''
    FERstart = 0
    classifyMoves = 0

    ## Head movement variables
    predictHeadMov = 3
    HeadMov = []
    HMCclass = ''
    detectedFaces = []
    mark = []

    notDone = 0
    progressStatus = [0,0]

    while notDone in progressStatus: # This waits for each module to finsih
        if camera == 'kinect':
            frames = listener.waitForNewFrame()

            color = frames["color"]
            color = color.asarray()
            color = cv2.resize(color, (int(873), int(491)))
            color = color[0:480, 150:790]
            color = np.delete(color, np.s_[3::], 2)

        elif camera == 'webcam':
            # Capture frame-by-frame
            ret, color = video_capture.read()

        elif camera == 'video':
            # Capture frame-by-frame
            ret, color = video_capture.read()

        ## Detect facial expression
        predictExpNums = [1,2]
        if predictExp == 0:
            while predictExp not in predictExpNums:
                predictExp = int(raw_input("\nPress 1 to detect Facial Expression or press 2 to do Head Movement classification."))
            if predictExp == 1:
                predictExp = 1
                print "------ Facial Expression Recognition ------"
            elif predictExp == 2:
                predictHeadMov = 0
                progressStatus[0] = 1

            FERstart = time.time()

        elif predictExp == 1:
            currentTime = time.time()
            if currentTime - FERstart < 10:
                FER_prediction.append(facialExpressionRecogntion(color, faceExpModel, faceCascade))
            else:
                predictExp = 2
                FER_prediction = filter(None, FER_prediction)
                FERclass = mostCommon(FER_prediction)
                FERclass = FERclass[2:7]
                predictHeadMov = 0
                if FERclass == '':
                    print("Did not detect an expression, try again.")
                    predictExp = 0
                else:
                    progressStatus[0] = 1
                    print "Detected expression: " + str(FERclass)
                    print "Progress: FER DONE"
                # else:
                #     #cv2.imwrite("sample_frame.png", color)
                #     break

        ## Detect head movement class
        if predictHeadMov == 0:
            predictHeadMov = int(raw_input("\nPress 1 to do Head Movement classification or 2 to skip."))
            if predictHeadMov == 1:
                predictHeadMov = 1
                print "------ Head Movement Classification ------"
                moveTime = int(raw_input("How many seconds should I record your movement?"))
                #moveTime = 2
            else:
                predictHeadMov = 2
            timer = time.time()
            previousTime1 = timer
            previousTime2 = timer

        if predictHeadMov == 1:
            print color.shape
            color = color[0:480, 0:480]
            color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            cv2.imshow("", color)
            raw_input()
            print color.shape
            # Feed frame to image queue.
            img_queue.put(color)

            #pdb.set_trace()

            # Get face from box queue.
            facebox = box_queue.get()
            print color.shape

            if facebox is not None:
                # Detect landmarks from image of 128x128.
                face_img = color[facebox[1]: facebox[3],
                                 facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                marks = mark_detector.detect_marks(face_img)

                #Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color, headPoseDirection, (20,70), font, 1, (0,255,0), 4)

                # # Get average position of nose
                noseMarksTemp = []
                noseMarksTemp.append(marks[30][0])
                noseMarksTemp.append(marks[30][1])
                noseMarks[0] = noseMarksTemp

                for i in range(9, 0, -1):
                    noseMarks[i] = noseMarks[i-1]

                # Get the direction of head movement
                headPoseDirection = calculateDirection(noseMarks)

                directionArray.append(headPoseDirection)

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
                    HMCclass = mostCommon(directionArray)

                    classifyMoves = 2
                    directionArray = []
                    previousTime1 = currentTime

                    progressStatus[1] = 1
                    print "Progress: HMC DONE"
            else:
                print "Did not detect a face"
        elif predictHeadMov == 2:
            progressStatus[1] = 1
            print "Skipped Head Movement Estimation."
            break

        # if notDone in progressStatus and predictHeadMov == 2 and predictExp == 2:
        #     print "You seem to have skipped one or more tasks."
        #     inpt = ''
        #     while inpt == '':
        #         inpt = raw_input("To do FER press 1 and to do HMC press 2...")
        #         if input == '1':
        #             predictExp = 1
        #         elif input == '2':
        #             predictHeadMov


        cv2.imshow("", color)
        if camera == 'kinect':
            listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    if camera == 'kinect':
        listener.release(frames)
        device.stop()
        device.close()
    elif camera == 'webcam' or camera == 'video':
        video_capture.release()

    cv2.destroyAllWindows()

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

    print "---------------- RESULT ----------------"

    if FERclass != '':
        print "Detected facial expression: " + str(FERclass)
    else:
        print "Did not detect any expression."

    if HMCclass != '':
        print "Detected head movement: " + str(HMCclass)
    else:
        print "Did not detect a head movement."

    print "----------------------------------------"

    intentionClassification = [FERclass, HMCclass]

    return intentionClassification


intentions = main()

print "Intentions: "
print "Facial Expression: " + intentions[0]
print "Head movement: " + intentions[1]
