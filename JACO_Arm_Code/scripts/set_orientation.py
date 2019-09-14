#!/usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca
# BRL 2018
#
#
# Generate Orientation Grid Functions


# import rospy
import numpy as np
import csv

""" Global variable """
RIGHT = 1
TOP = 2
#side_dim = 0.8
#JACO_position = 3
#grid_resolution = 0.1
#item_position = [0.20, -0.65, 0.09]      # X,Y,Z
#goal_position = [-0.20, -0.65, 0.09]


def set_side_dim(table_coords):
    side_dim = 0

    if (table_coords[3] == table_coords[4] == table_coords[5] == table_coords[6]):
        side_dim = table_coords[3]

    return side_dim


def set_JACO_position(JACO_xy, side_dim):
    JACO_position = 0

    if 0 < JACO_xy[1] < (side_dim/3):
        JACO_position = 3
    elif (side_dim/3) < JACO_xy[1] < (2*side_dim/3):
        JACO_position = 2
    elif (2*side_dim/3) < JACO_xy[1] < side_dim:
        JACO_position = 1

    return JACO_position


def convert_reference(coords_CAMERAref, JACO_coords):
    coords_JACOref = []

    coords_JACOref.append(coords_CAMERAref[1] - JACO_coords[1])
    coords_JACOref.append(JACO_coords[0] - coords_CAMERAref[0])
    coords_JACOref.append(coords_CAMERAref[2])

    return coords_JACOref


# Divide the grid into sections
def generate_grid(JACO_position, side_dim, grid_resolution):
    l = side_dim
    l_vector = []
    x_axis = []
    y_axis = []

    while l > grid_resolution:
        l = l / 2

    i = 0
    while i <= side_dim:
        l_vector.append(round(i, 2))
        i = i + l

    if JACO_position == 1:

        # higher side of the table
        i = 0
        j = 1
        while i < len(l_vector):
            y_axis.append(-l_vector[i])
            x_axis.append(-l_vector[i])
            i = i + 1
        print(x_axis)
        print(y_axis)

    elif JACO_position == 2:

        # middle of the table side
        i = 0
        j = 1
        while i < len(l_vector):
            y_axis.append(-l_vector[i])
            if i < (len(l_vector)//2):
                x_axis.append(-l_vector[(len(l_vector)//2) - i])
            elif i == (len(l_vector)/2):
                x_axis.append(0)
            elif i > (len(l_vector)//2):
                x_axis.append(-x_axis[(len(l_vector)//2) - j])
                j = j + 1
            i = i + 1
        print(x_axis)
        print(y_axis)

    elif JACO_position == 3:

        # lower side of the table
        i = 0
        j = 1
        while i < len(l_vector):
            y_axis.append(-l_vector[i])
            x_axis.append(l_vector[i])
            i = i + 1
        print(x_axis)
        print(y_axis)

    return x_axis, y_axis


# Generate and match the csv file with the grid (RIGHT HOLDING)
def gen_orientation_grid(x_axis, y_axis, grabbing):
    file = 0
    print "Hello"
    if grabbing == RIGHT:
        file = "/home/project/my_jaco_package/src/goToPosition/table_right.csv"
    elif grabbing == TOP:
        file = "/home/project/my_jaco_package/src/goToPosition/table_top.csv"

    #bag = open("table_right.csv", "r")
    bag = open(file, "r")

    reader = csv.reader(bag)
    grid = np.zeros((len(x_axis), len(y_axis), 3))

    for line in reader:
        X = float(line[0])
        Y = float(line[1])

        i = 0
        j = 0
        while i < len(x_axis):
            if x_axis[i] < X < x_axis[i+1]:
                while j < len(y_axis):
                    if y_axis[j] > Y > y_axis[j + 1]:
                        grid[i][j][0] = line[3]
                        grid[i][j][1] = line[4]
                        grid[i][j][2] = line[5]
                        break
                    j = j + 1
                break
            i = i + 1
    print (grid)
    return grid


# Generate and match the csv file with the grid (TOP HOLDING)
def gen_orientation_grid_top(x_axis, y_axis):
    bag = open("table_top.csv", "r")
    #bag = open(r'/home/project/')

    reader = csv.reader(bag)
    grid = np.zeros((len(x_axis), len(y_axis), 3))

    for line in reader:
        X = float(line[0])
        Y = float(line[1])

        i = 0
        j = 0
        while i < len(x_axis):
            if x_axis[i] < X < x_axis[i+1]:
                while j < len(y_axis):
                    if y_axis[j] > Y > y_axis[j + 1]:
                        grid[i][j][0] = line[3]
                        grid[i][j][1] = line[4]
                        grid[i][j][2] = line[5]
                        break
                    j = j + 1
                break
            i = i + 1
    print (grid)
    return grid


# Depending on the area where the item is detected, set thetaX, thetaY, thetaZ (ORIENTATION)
def set_orientation(x_axis, y_axis, grid, item_position):
    item_orientation = []

    for i in range(0, len(x_axis)):
        if x_axis[i] < item_position[0] < x_axis[i + 1]:
            for j in range(0, len(y_axis)):
                if y_axis[j] > item_position[1] >y_axis[j+1]:
                    item_orientation.append((grid[i][j][0]))
                    item_orientation.append((grid[i][j][1]))
                    item_orientation.append((grid[i][j][2]))
                    break
            break

    item = item_position + item_orientation
    print (item)
    return item
