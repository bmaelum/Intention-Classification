#!/usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca, Roberto Geografo
# BRL - Summer 2017
#
#
# Main file

import rospy
import numpy as np

from pose_action_client import send_coords, getcurrentCartesianCommand
from joints_action_client import send_joints
from fingers_action_client import main_fingers
from adjust import main_adjust, adjust_fingers_closing_size, adjust_fingers_closing
from set_orientation import set_side_dim, set_JACO_position, convert_reference, generate_grid, gen_orientation_grid, set_orientation
from set_handle import set_handle_orientation, set_grabbing, rotation_angle

import intentionClassifier
import copy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dlib
import cv2

""" Global variable """
RIGHT = 1
TOP = 2
ABSOLUTE = 1
RELATIVE = 2
grid_resolution = 0.1


if __name__ == '__main__':

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

    # Comment out if running only this from IDE
    rospy.init_node('main_file', anonymous=True)

    print "-- Intention Classification JACO Movement --"

    item = []
    goal = []

    # From the CAMERA I will get:
    # JACO_coord, table_coords, item_position, handle_position, goal position
    # ALL WRT THE CAMERA'S ORIGIN
    JACO_coords = [0.05, 0.4]  # centre of JACO's base
    table_coords = [0, 0, 0, 0.8, 0.8, 0.8, 0.8, 0]  ##4 corners

    pos_dict = {
                'C'         : [0.020, -0.574, 0.0587, -3.118, -0.026, -1.556], # from terminal
                'N'         : [0.006, -0.466, 0.0476, -3.118, -0.009, -1.556],
                'S'         : [0.017,  -0.660, 0.066, -3.118, 0.048, -1.556],
                'E'         : [0.222, -0.559, 0.041, 3.134, 0.075, -1.556],
                'W'         : [-0.202, -0.573, 0.040, 3.052, 0.042, -1.556],
                'NW'         : [-0.184, -0.468, 0.058, 3.095, 0.049, -1.556],
                'NE'         : [0.231, -0.449, 0.054, 3.105, 0.079, -1.556],
                'SW'         : [-0.194, -0.645, 0.047, 3.131, 0.061, -1.474],
                'SE'         : [0.234, -0.646, 0.0683, 3.119, 0.066, -1.514]
    }

    hoverPos = copy.deepcopy(pos_dict)
    hoverPos['C'][2] += 0.2
    hoverPos['N'][2] += 0.2
    hoverPos['S'][2] += 0.2
    hoverPos['E'][2] += 0.2
    hoverPos['W'][2] += 0.2
    hoverPos['SE'][2] += 0.2
    hoverPos['SW'][2] += 0.15
    hoverPos['NE'][2] += 0.2
    hoverPos['NW'][2] += 0.2

    f_closing = adjust_fingers_closing_size(0.035)  # closing
    f_opening = adjust_fingers_closing_size(0.200)  # closing

    def getReadyPosition():
        center_after_pickup_position = [0.032, -0.573, 0.25, 3.108, 0.069, -1.556]
        send_coords(center_after_pickup_position, ABSOLUTE)

        #testRotateGripper = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        #send_coords(testRotateGripper, RELATIVE)

        main_fingers(f_opening)

    #getReadyPosition()

    def getDir(move):
        if move == 'north':
            return 'N'
        elif move == 'south':
            return 'S'
        elif move == 'east':
            return 'E'
        elif move == 'west':
            return 'W'
        elif move == 'north east':
            return 'NE'
        elif move == 'north west':
            return 'NW'
        elif move == 'south east':
            return 'SE'
        elif move == 'south west':
            return 'SW'

    def goToAckPos():
        main_fingers(f_opening)
        acknowledgementPos = [0.027, -0.472, 0.53, 1.614, 0.0059, 0.171]
        send_coords(acknowledgementPos, ABSOLUTE)
        main_fingers(f_closing)
        main_fingers(f_opening)
        main_fingers(f_closing)
        main_fingers(f_opening)

    initialize = 1
    old_box_pos = 'C' # current position of box at all times
    new_box_pos = ''
    while True:
        if initialize == 1:
            print "Make sure object is in center position!"
            inputCommand = int((raw_input("\nPress 1 to proceed or 2 to cancel...")))
            if inputCommand == 1:
                goToAckPos()
                facialExpression = intentionClassifier.intentClass(1, faceExpModel, faceCascade, detector, predictor)
                print "Detected expression: " + facialExpression
                if facialExpression == 'happy':
                    initialize = 0
                else:
                    print "Initialization failed..."
                    print "Try again..."
            else:
                break

        else:
            print "Moving to ready position..."
            getReadyPosition()
            facialExpression, headMovement = intentionClassifier.intentClass(3, faceExpModel, faceCascade, detector, predictor)
            print "Facial Expression: " + facialExpression
            print "Head movement: " + headMovement
            #f_opening = adjust_fingers_closing_size(0.200)  # closing
            #main_fingers(f_opening)
            # Do intentionClassification
            new_box_pos = getDir(headMovement)
            #new_box_pos = raw_input("Write your desired drop point(C,N,S,E,W,NE,NW,SE,SW)...")
            print "Desired drop point is " + str(new_box_pos)

            if new_box_pos == 'X':
                break

            elif new_box_pos == 'I':
                initialize = 1

            elif new_box_pos == old_box_pos:
                print "Box is already at position " + str(new_box_pos)

            else:
                # Pick up box at old box position
                #send_coords([0.0, 0.0, 0.10, 0.0, 0.0, 0.0], RELATIVE)
                send_coords(hoverPos[old_box_pos], ABSOLUTE)
                send_coords(pos_dict[old_box_pos], ABSOLUTE)
                main_fingers(f_closing)
                #send_coords([0.0, 0.0, 0.10, 0.0, 0.0, 0.0], RELATIVE)

                print hoverPos[new_box_pos]
                print pos_dict[new_box_pos]

                # Drop box in new position
                send_coords(hoverPos[new_box_pos], ABSOLUTE)
                send_coords(pos_dict[new_box_pos], ABSOLUTE)
                main_fingers(f_opening)
                send_coords(hoverPos[new_box_pos], ABSOLUTE)
                old_box_pos = new_box_pos
