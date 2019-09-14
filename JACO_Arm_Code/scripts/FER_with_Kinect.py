import cv2
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
#from estimate_head_pose import headPoseEstimation
import estimate_head_pose
from expressionRecognition import detectFace, cropFaces, displayCrop, predictImg

## --------------- MAIN ------------------
def main():
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
    print("Packet pipeline:", type(pipeline).__name__)

    # Create and set logger
    #logger = createConsoleLogger(LoggerLevel.Debug)
    #setGlobalLogger(logger)

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
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

    cascPath = "/home/bjornar/MScDissertation/TrainingData/FaceDetection/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    #faceExpModel = tf.keras.models.load_model("/home/bjornar/MScDissertation/createModel/ModelCreatedFromWeights.hdf5")

    faceExpModel = tf.keras.models.load_model("/home/bjornar/ML_models/FER/Good models(80+)/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")
    print "To detect facial expression press 'e'"
    expRecognition = 0

    while True:
        frames = listener.waitForNewFrame()

        color = frames["color"]
        #color = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))
        color = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))

        #movementClass = estimate_head_pose.headPoseEstimation(color)
        #print movementClass
        #cv2.imshow("color", color)
        if cv2.waitKey(5) == ord('e'):
            expRecognition = 1
            print "Detecting facial expression..."

        if expRecognition:
            gray, faces = detectFace(color, faceCascade)

            prediction = []

            if len(faces) > 0:
                croppedFace = cropFaces(color, faces)
                #displayCrop(croppedFace)
                for face in croppedFace:
                    #cv2.imwrite("croppedface.png", face)
                    prediction.append(predictImg(face, faceExpModel))
                #displayFace(faces, color)
                #cv2.waitKey(0)
                for (x,y,w,h) in faces:
                        cv2.rectangle(color, (x,y), (x+w, y+h), (0,255,0), 2)

                #displayFace(faces, color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color, prediction[0], (20,70), font, 1, (0,255,0), 4)
                cv2.imshow("", color)

        else:
            cv2.imshow("", color)


        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    device.stop()
    device.close()

main()
