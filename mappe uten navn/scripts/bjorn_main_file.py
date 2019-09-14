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

from pose_action_client import send_coords
from joints_action_client import send_joints
from fingers_action_client import main_fingers
from adjust import main_adjust, adjust_fingers_closing_size, adjust_fingers_closing
from set_orientation import set_side_dim, set_JACO_position, convert_reference, generate_grid, gen_orientation_grid, set_orientation
from set_handle import set_handle_orientation, set_grabbing, rotation_angle


""" Global variable """
RIGHT = 1
TOP = 2
ABSOLUTE = 1
RELATIVE = 2
grid_resolution = 0.1

if __name__ == '__main__':

    # Comment out if running only this from IDE
    rospy.init_node('main_file', anonymous=True)

    item = []
    goal = []

    # From the CAMERA I will get:
    # JACO_coord, table_coords, item_position, handle_position, goal position
    # ALL WRT THE CAMERA'S ORIGIN
    JACO_coords = [0.05, 0.4]  # centre of JACO's base
    table_coords = [0, 0, 0, 0.8, 0.8, 0.8, 0.8, 0]  ##4 corners
    item_position = [0.68, 0.64, 0.08]
    handle_position = [0.70, 0.64, 0.08]
    goal_position = [0.68, 0.10, 0.08]

    # Set table size from 4 corners coordinates
    side_dim = set_side_dim(table_coords)

    # Set JACO position, given the base coordinates
    JACO_position = set_JACO_position(JACO_coords, side_dim)

    # Convert all the coordinates wrt to the JACO arm reference
    item_position = convert_reference(item_position, JACO_coords)
    print(item_position)
    handle_position = convert_reference(handle_position, JACO_coords)
    print(handle_position)
    goal_position = convert_reference(goal_position, JACO_coords)
    print(goal_position)

    # Generate grids
    [x_axis, y_axis] = generate_grid(JACO_position, side_dim, grid_resolution)
    grid_right = gen_orientation_grid(x_axis, y_axis, RIGHT)
    grid_top = gen_orientation_grid(x_axis, y_axis, TOP)

    # Set handle orientation
    handle_direction = set_handle_orientation(item_position, handle_position)

    # Decide how to grab it (either from the right, or from the top > get it closer > from the right)
    method = set_grabbing(handle_direction, JACO_position)

    if method == RIGHT:   #from the RIGHT
        item = set_orientation(x_axis, y_axis, grid_right, item_position)
        goal = set_orientation(x_axis, y_axis, grid_right, goal_position)

    elif method == TOP:   #from the TOP
        angle = rotation_angle(handle_direction)
        angle_v = [0.0, 0.0, 0.0, 0.0, 0.0, angle]

        item = set_orientation(x_axis, y_axis, grid_top, item_position)

        item_UP = item
        item_UP[2] = item[2] + 0.1
        send_coords(item_UP, ABSOLUTE)

        send_coords([0.0, 0.0, -0.1, 0.0, 0.0, 0.0], RELATIVE)

        f_closing = adjust_fingers_closing_size(0.100)  # closing
        main_fingers(f_closing)

        send_joints(angle_v)
        main_fingers([0, 0, 0])

        item_right = set_orientation(x_axis, y_axis, grid_right, item_position)
        send_coords(item_right, ABSOLUTE)

        main_fingers(f_closing)
