#! /usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca, Roberto Geografo
# BRL - Summer 2017
#
#
# Kinova Fingers control file


import rospy

import actionlib
import kinova_msgs.msg


""" Global variable """
prefix = 'j2n6s300_'  #Jaco2, 6DOF, service, 3fingers
finger_maxDist = 18.9/2/1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentFingerPosition = [0.0, 0.0, 0.0]


def gripper_client(finger_positions):
    """Send a gripper goal to the action server."""
    action_address = '/' + prefix + 'driver/fingers_action/finger_positions'

    client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.SetFingersPositionAction)
    client.wait_for_server()

    goal = kinova_msgs.msg.SetFingersPositionGoal()
    goal.fingers.finger1 = float(finger_positions[0])
    goal.fingers.finger2 = float(finger_positions[1])
    goal.fingers.finger3 = float(finger_positions[2])
    client.send_goal(goal)
    if client.wait_for_result(rospy.Duration(10.0)):
        return client.get_result()
    else:
        client.cancel_all_goals()
        rospy.WARN('        the gripper action timed-out')
        return None


def getCurrentFingerPosition(prefix_):
    # wait to get current position
    topic_address = '/' + prefix_ + 'driver/out/finger_position'
    rospy.Subscriber(topic_address, kinova_msgs.msg.FingerPosition, setCurrentFingerPosition)
    rospy.wait_for_message(topic_address, kinova_msgs.msg.FingerPosition)
    print 'obtained current finger position '


def setCurrentFingerPosition(feedback):
    global currentFingerPosition
    currentFingerPosition[0] = feedback.finger1
    currentFingerPosition[1] = feedback.finger2
    currentFingerPosition[2] = feedback.finger3


def fingers(finger_value_):
    global currentFingerPosition

    finger_turn_command = [x / 1000 * finger_maxTurn / finger_maxDist for x in finger_value_]

    finger_turn_absolute_ = finger_turn_command

    finger_turn_ = finger_turn_absolute_
    finger_meter_ = [x * finger_maxDist / finger_maxTurn for x in finger_turn_]
    finger_percent_ = [x / finger_maxTurn * 100.0 for x in finger_turn_]

    return finger_turn_, finger_meter_, finger_percent_


def main_fingers(finger_value_):

    getCurrentFingerPosition(prefix)

    finger_turn, finger_meter, finger_percent = fingers(finger_value_)

    try:

        positions_temp1 = [max(0.0, numb) for numb in finger_turn]
        positions_temp2 = [min(numb, finger_maxTurn) for numb in positions_temp1]
        positions = [float(numb) for numb in positions_temp2]

        print('Sending finger position ...')
        result1 = gripper_client(positions)
        print('Finger position sent!')

    except rospy.ROSInterruptException:
        print('program interrupted before completion')

    return 1

