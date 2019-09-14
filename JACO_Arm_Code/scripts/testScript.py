#!/usr/bin/env python
#
# MSc ROBOTICS: NON-VERBAL HUMAN ROBOT INTERACTION
# Supervisor: Khemraj Emrith
# Student: Bjoernar Maelum
# BRL - Summer 2018
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
import sys

""" Global variable """
RIGHT = 1
TOP = 2
ABSOLUTE = 1
RELATIVE = 2
grid_resolution = 0.1

# # Load Facial Expression Recognition trained model
# print "- Loading FER model... -"
# #faceExpModel = tf.keras.models.load_model("/home/bjornar/ML_models/FER/Good models(80+)/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")
# faceExpModel = tf.keras.models.load_model("/home/project/Bjorn/tf_keras_weights_ninthRev-88percent/tf_keras_weights_ninthRev.hdf5")
#
# # Load Face Cascade for Face Detection
# print "- Loading Face Cascade for Face Detection... -"
# cascPath = "/home/project/IntentionClassification-Repository/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/home/project/my_jaco_package/src/bjornPkg/scripts/shape_predictor_68_face_landmarks.dat")

#print sys.argv

if __name__ == '__main__':

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
                'C'         : [-0.016, -0.574, 0.0587, -3.118, -0.026, -1.556], # from terminal
                'N'         : [-0.016, -0.466, 0.0587, -3.118, -0.009, -1.556],
                'S'         : [-0.016,  -0.660, 0.0587, -3.118, 0.048, -1.556],
                'E'         : [0.20, -0.559, 0.0587, 3.134, 0.075, -1.556],
                'W'         : [-0.230, -0.553, 0.0587, 3.052, 0.042, -1.556],
                'NW'        : [-0.230, -0.448, 0.0587, 3.095, 0.049, -1.556],
                'NE'        : [0.20, -0.454, 0.0587, 3.127, 0.059, -1.607],#[0.20, -0.449, 0.054, 3.105, 0.079, -1.556],
                'SW'        : [-0.230, -0.645, 0.0587, 3.131, 0.061, -1.474],
                'SE'        : [0.20, -0.646, 0.0587, 3.119, 0.066, -1.514],
                'home'      : [0.212, -0.257, 0.509, 1.637, 1.113, -0.134]
    }

    hoverPos = copy.deepcopy(pos_dict)
    hoverPos['C'][2] += 0.15
    hoverPos['N'][2] += 0.15
    hoverPos['S'][2] += 0.15
    hoverPos['E'][2] += 0.15
    hoverPos['W'][2] += 0.15
    hoverPos['SE'][2] += 0.15
    hoverPos['SW'][2] += 0.15
    hoverPos['NE'][2] += 0.15
    hoverPos['NW'][2] += 0.15

    f_closing = adjust_fingers_closing_size(0.035)  # closing
    f_opening = adjust_fingers_closing_size(0.200)  # closing

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
        else:
            return 'C'

    def getReadyPosition():
        print "Moving to ready position..."

        center_after_pickup_position = [0.016, -0.573, 0.25, 3.108, 0.069, -1.556]
        send_coords(center_after_pickup_position, ABSOLUTE)

        main_fingers(f_opening)

    def ackGesture():
        main_fingers(f_opening)
        acknowledgementPos = [0.035, -0.663, 0.342, 1.596, 0.023, 1.527]
        send_coords(acknowledgementPos, ABSOLUTE)
        main_fingers(f_closing)
        main_fingers(f_opening)
        main_fingers(f_closing)
        main_fingers(f_opening)

    def getAck():
        facialExpression = intentionClassifier.intentClass(1, faceExpModel, faceCascade, detector, predictor)
        print "Detected: " + str(facialExpression)
        if facialExpression == 'happy':
            return 1
        else:
            return 0

    def thumbsUp():
        main_fingers(f_opening)

        acknowledgementPos = [-0.016, -0.663, 0.342, 1.596, 0.023, 1.527]
        send_coords(acknowledgementPos, ABSOLUTE)

        main_fingers(f_closing)
        main_fingers(f_opening)
        main_fingers(f_closing)

    def homePosition():
        print "Go to Home Position"
        # Move to home position
        send_coords(pos_dict['home'], ABSOLUTE)
        main_fingers(f_opening)

    def endInteraction():
        print "End interaction"
        # move box to default position

        # End of interaction gesture

        # Move to home position

    def pickAndPlace(old_pos, new_pos):
        # Pick up box at old box position
        #send_coords([0.0, 0.0, 0.10, 0.0, 0.0, 0.0], RELATIVE)
        send_coords(hoverPos[old_pos], ABSOLUTE)
        send_coords(pos_dict[old_pos], ABSOLUTE)
        main_fingers(f_closing)
        #send_coords([0.0, 0.0, 0.10, 0.0, 0.0, 0.0], RELATIVE)

        #print hoverPos[new_box_pos]
        #print pos_dict[new_box_pos]

        # Drop box in new position
        send_coords(hoverPos[new_pos], ABSOLUTE)
        send_coords(pos_dict[new_pos], ABSOLUTE)
        main_fingers(f_opening)
        #send_coords(hoverPos[new_pos], ABSOLUTE)
        #old_pos = new_pos

    def test2_1():
        print "Moving to ready position..."
        homePosition()
        getReadyPosition()

        center_after_pickup_position = [-0.016, -0.573, 0.25, 3.108, 0.069, -1.556]
        send_coords(center_after_pickup_position, ABSOLUTE)

    def test2_2():
        print "Perform acknowledgement gesture"
        homePosition()
        ackGesture()

    def test2_3():
        print "Move object from C to N"
        homePosition()
        pickAndPlace("C", "N")

    def test2_4():
        print "Move object from N to C"
        homePosition()
        pickAndPlace("N", "C")

    def test2_5():
        print "Move object from C to S"
        homePosition()
        pickAndPlace("C", "S")

    def test2_6():
        print "Move object from S to C"
        homePosition()
        pickAndPlace("S", "C")

    def test2_7():
        print "Move object from C to W"
        homePosition()
        pickAndPlace("C", "W")

    def test2_8():
        print "Move object from W to C"
        homePosition()
        pickAndPlace("W", "C")

    def test2_9():
        print "Move object from C to E"
        homePosition()
        pickAndPlace("C", "E")

    def test2_10():
        print "Move object from E to C"
        homePosition()
        pickAndPlace("E", "C")

    def test2_11():
        print "Move object from C to NW"
        homePosition()
        pickAndPlace("C", "NW")

    def test2_12():
        print "Move object from NW to C"
        homePosition()
        pickAndPlace("NW", "C")

    def test2_13():
        print "Move object from C to NE"
        homePosition()
        pickAndPlace("C", "NE")

    def test2_14():
        print "Move object from NE to C"
        homePosition()
        pickAndPlace("NE", "C")

    def test2_15():
        print "Move object from C to SW"
        homePosition()
        pickAndPlace("C", "SW")

    def test2_16():
        print "Move object from SW to C"
        homePosition()
        pickAndPlace("SW", "C")

    def test2_17():
        print "Move object from C to SE"
        homePosition()
        pickAndPlace("C", "SE")

    def test2_18():
        print "Move object from SE to C"
        homePosition()
        pickAndPlace("SE", "C")


    def endGesture():
        # send_coords(hoverPos[old_pos], ABSOLUTE)
        # send_coords(pos_dict[old_pos], ABSOLUTE)
        # main_fingers(f_closing)

        objectToHuman_coords = [-0.037, -0.675, 0.264, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.164, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.364, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.264, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        main_fingers(f_closing)
        main_fingers(f_opening)
        main_fingers(f_closing)

    def endInteraction(old_pos):
        print "End interaction"

        if old_pos != 'C':
            # move box to default position
            _ = pickAndPlace(old_pos, 'C')

        # End of interaction gesture
        endGesture()

        # Move to home position
        homePosition()

    def handObjecToHuman(old_pos):
        send_coords(hoverPos[old_pos], ABSOLUTE)
        send_coords(pos_dict[old_pos], ABSOLUTE)
        main_fingers(f_closing)

        objectToHuman_coords = [-0.037, -0.675, 0.264, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

    def endGesture():
        objectToHuman_coords = [-0.037, -0.675, 0.264, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.164, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.364, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        objectToHuman_coords = [-0.037, -0.675, 0.264, 1.436, -0.063, -1.338]
        send_coords(objectToHuman_coords, ABSOLUTE)

        main_fingers(f_closing)
        main_fingers(f_opening)
        main_fingers(f_closing)

    def runTests_2():
        test2_1()
        test2_2()
        test2_3()
        test2_4()
        test2_5()
        test2_6()
        test2_7()
        test2_8()
        test2_9()
        test2_10()
        test2_11()
        test2_12()
        test2_13()
        test2_14()
        test2_15()
        test2_16()
        test2_17()
        test2_18()
        pickAndPlace('C', 'S')
        handObjecToHuman('S')
        endInteraction('S')


    runTests_2()
    #homePosition()
    #getReadyPosition()
    #endGesture()

    homePosition()
