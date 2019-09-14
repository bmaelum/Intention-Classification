#! /usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca, Roberto Geografo
# BRL - Summer 2017
#
#
# Kinova Joints control file

import rospy

import actionlib
import kinova_msgs.msg

import math


""" Global variable """
arm_joint_number = 6
prefix = 'j2n6s300_'
finger_maxDist = 18.9/2/1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentJointCommand = [] # number of joints is defined in __main__

def joint_angle_client(angle_set):
    """Send a joint angle goal to the action server."""
    action_address = '/' + prefix + 'driver/joints_action/joint_angles'
    client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.ArmJointAnglesAction)
    client.wait_for_server()

    goal = kinova_msgs.msg.ArmJointAnglesGoal()

    goal.angles.joint1 = angle_set[0]
    goal.angles.joint2 = angle_set[1]
    goal.angles.joint3 = angle_set[2]
    goal.angles.joint4 = angle_set[3]
    goal.angles.joint5 = angle_set[4]
    goal.angles.joint6 = angle_set[5]
    goal.angles.joint7 = angle_set[6]

    client.send_goal(goal)
    if client.wait_for_result(rospy.Duration(20.0)):
        return client.get_result()
    else:
        print('        the joint angle action timed-out')
        client.cancel_all_goals()
        return None


def getcurrentJointCommand(prefix_):
    # wait to get current position
    topic_address = '/' + prefix_ + 'driver/out/joint_command'
    rospy.Subscriber(topic_address, kinova_msgs.msg.JointAngles, setcurrentJointCommand)
    rospy.wait_for_message(topic_address, kinova_msgs.msg.JointAngles)
    print 'position listener obtained message for joint position. '


def setcurrentJointCommand(feedback):
    global currentJointCommand

    currentJointCommand_str_list = str(feedback).split("\n")
    for index in range(0,len(currentJointCommand_str_list)):
        temp_str=currentJointCommand_str_list[index].split(": ")
        currentJointCommand[index] = float(temp_str[1])


def unitParser(joint_value):
    """ Argument unit """
    global currentJointCommand

    joint_degree_command = joint_value
    joint_degree_absolute_ = [joint_degree_command[i] + currentJointCommand[i] for i in range(0, len(joint_value))]
    joint_degree = joint_degree_absolute_

    return joint_degree


def send_joints(joint_value_):
    global currentJointCommand

    currentJointCommand = [0] * 7
    getcurrentJointCommand(prefix)

    joint_degree = unitParser(joint_value_)

    positions = [0]*7
    try:
        for i in range(0, arm_joint_number):
            positions[i] = joint_degree[i]

        result = joint_angle_client(positions)
        print('Joint parameters sent!')


    except rospy.ROSInterruptException:
        print "program interrupted before completion"

    return 1