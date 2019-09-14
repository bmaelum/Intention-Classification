#!/usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca
# BRL 2018
#
#
# Manage Mugs Handle orientation


# import rospy
import numpy as np


""" Global variable """
NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4


def set_handle_orientation(item_position, handle_position):
    direction = 0

    if (handle_position[0] == item_position[0]) and (handle_position[1] < item_position[1]):
        direction = NORTH
    elif (handle_position[0] < item_position[0]) and (handle_position[1] == item_position[1]):
        direction = EAST
    elif (handle_position[0] == item_position[0]) and (handle_position[1] > item_position[1]):
        direction = SOUTH
    elif (handle_position[0] > item_position[0]) and (handle_position[1] == item_position[1]):
        direction = WEST

    return direction


def set_grabbing(handle_direction, JACO_position):
    method = 0

    if handle_direction == EAST:
        method = 1    #from the RIGHT
    elif handle_direction == NORTH:
        method = 2   #from the TOP

    return method


def rotation_angle(handle_direction):
    angle = 0

    if handle_direction == NORTH:
        angle = -90
    elif handle_direction == EAST:
        angle = 0
    elif handle_direction == SOUTH:
        angle = 90
    elif handle_direction == WEST:
        angle = 180

    return angle