#! /usr/bin/env python
#
#
# MULTIMODAL HUMAN-ROBOT INTERACTION FOR ASSISTED LIVING
# Supervisor: Khemraj Emrith
# Students: Miriam Cristofoletti, Mihai Anca, Roberto Geografo
# BRL - Summer 2017
#
#
# Adjusting coordinates file


import math


# Find item's centroid
def findCentroid(XYcoord_):

    c_x = (XYcoord_[2] - XYcoord_[0]) / 2 + XYcoord_[0]
    #print c_x
    c_y = (XYcoord_[3] - XYcoord_[1]) / 2 + XYcoord_[1]
    #print c_y
    centre = []
    centre.append(c_x)
    centre.append(c_y)
    print centre

    return centre


# Set ThetaX, ThetaY, ThetaZ depending on the position
def setOrientation(centre_):

    # Areas list (see sketch on logbook, p43) ARM GRABBING FROM THE RIGHT
    coord_1 = [-0.000848710537 , -0.628232240677, -0.01407879666798, 1.68766689301,  1.05899548531,  -0.279898703098] # top left corner
    coord_2 = [-0.351131439209 , -0.622758150101, -0.01436397691071, 1.62944030762,  0.349061191082, -0.198951348662] # top middle
    coord_3 = [-0.612733840942 , -0.528163194656, -0.01428429697827, 1.63678646088, -0.493506163359, -0.14812579751 ] # top right corner
    coord_4 = [ 0.0006842315197, -0.333499789238, -0.01497377754375, 1.6323775053 ,  0.366252958775, -0.202159389853] # middle left
    coord_5 = [-0.378755629063 , -0.330823779106, -0.01469458461925, 1.63624727726, -0.487316995859, -0.149051710963] # middle
    coord_6 = [-0.676275253296 , -0.330867052078, -0.01445097484812, 1.64999520779, -0.74950170517,  -0.125600755215] # middle right
    coord_7 = [-0.345254719257 ,  0.008651643991, -0.01479751466587, 1.74077224731, -1.22614777088,  -0.020272538066] # bottom middle
    coord_8 = [-0.70036649704  ,  0.008427843452, -0.0143132894896 , 1.73299837112, -1.21400690079,  -0.028210818767] # bottom right corner

    #Item's centre should be with negative coordinates (see Jaco arm xyz axes)
    centre_[0] = -centre_[0]
    centre_[1] = -centre_[1]

    #Depending on the area, set z heigth, thetaX, thetaY, thetaZ
    if (centre_[0] < 0) and (centre_[0] > -0.15):
        if (centre_[1] < -0.55) and (centre_[1] > -0.66):
            #Area 1
            centre_.append(coord_1[2])
            centre_.append(coord_1[3])
            centre_.append(coord_1[4])
            centre_.append(coord_1[5])

        elif (centre_[1] < -0.43) and (centre_[1] > -0.56):
            #Area 1-4
            centre_.append(coord_1[2])
            centre_.append((coord_1[3] - coord_4[3]) / 2 + coord_4[3])
            centre_.append((coord_1[4] - coord_4[4]) / 2 + coord_4[4])
            centre_.append((coord_1[5] - coord_4[5]) / 2 + coord_4[5])

        elif (centre_[1] < -0.24) and (centre_[1] > -0.44):
            #Area 4
            centre_.append(coord_4[2])
            centre_.append(coord_4[3])
            centre_.append(coord_4[4])
            centre_.append(coord_4[5])

    elif (centre_[0] < -0.16) and (centre_[0] > -0.55):
        if (centre_[1] < -0.55) and (centre_[1] > -0.65):
            #Area 2
            centre_.append(coord_2[2])
            centre_.append(coord_2[3])
            centre_.append(coord_2[4])
            centre_.append(coord_2[5])

        elif (centre_[1] < -0.43) and (centre_[1] > -0.55):
            #Area 2-5
            centre_.append(coord_5[2])
            centre_.append((coord_5[3] - coord_2[3]) / 2 + coord_2[3])
            centre_.append((coord_2[4] - coord_5[4]) / 2 + coord_5[4])
            centre_.append((coord_5[5] - coord_2[5]) / 2 + coord_2[5])

        elif (centre_[1] < -0.24) and (centre_[1] > -0.44):
            # Area 5
            centre_.append(coord_5[2])
            centre_.append(coord_5[3])
            centre_.append(coord_5[4])
            centre_.append(coord_5[5])

        elif (centre_[1] < -0.15) and (centre_[1] > -0.25):
            # Area 5-7
            centre_.append(coord_7[2])
            centre_.append((coord_7[3] - coord_5[3]) / 2 + coord_5[3])
            centre_.append((coord_7[4] - coord_5[4]) / 2 + coord_5[4])
            centre_.append((coord_7[5] - coord_5[5]) / 2 + coord_5[5])

        elif (centre_[1] < 0) and (centre_[1] > -0.16):
            # Area 7
            centre_.append(coord_7[2])
            centre_.append(coord_7[3])
            centre_.append(coord_7[4])
            centre_.append(coord_7[5])

    elif (centre_[0] < -0.56) and (centre_[0] > -0.65):
        if (centre_[1] < -0.55) and (centre_[1] > -0.65):
            # Area 3
            centre_.append(coord_3[2])
            centre_.append(coord_3[3])
            centre_.append(coord_3[4])
            centre_.append(coord_3[5])

        elif (centre_[1] < -0.43) and (centre_[1] > -0.55):
            # Area 3-6
            centre_.append(coord_6[2])
            centre_.append((coord_6[3] - coord_3[3]) / 2 + coord_3[3])
            centre_.append((coord_3[4] - coord_6[4]) / 2 + coord_6[4])
            centre_.append((coord_6[5] - coord_3[5]) / 2 + coord_3[5])

        elif (centre_[1] < -0.24) and (centre_[1] > -0.44):
            # Area 6
            centre_.append(coord_6[2])
            centre_.append(coord_6[3])
            centre_.append(coord_6[4])
            centre_.append(coord_6[5])

        elif (centre_[1] < -0.15) and (centre_[1] > -0.25):
            # Area 6-8
            centre_.append(coord_8[2])
            centre_.append((coord_8[3] - coord_6[3]) / 2 + coord_6[3])
            centre_.append((coord_8[4] - coord_6[4]) / 2 + coord_6[4])
            centre_.append((coord_8[5] - coord_6[5]) / 2 + coord_6[5])

        elif (centre_[1] < 0) and (centre_[1] > -0.16):
            # Area 8
            centre_.append(coord_8[2])
            centre_.append(coord_8[3])
            centre_.append(coord_8[4])
            centre_.append(coord_8[5])

    return centre_


# Main adjusting function
def main_adjust(XY_coord):

    # Find item's centroid
    XYcentre = findCentroid(XY_coord)

    # Set ThetaX, ThetaY, ThetaZ depending on the position
    EEcentre = setOrientation(XYcentre)

    print EEcentre

    return EEcentre

# 0.154 0.342 0.256 0.443 1


# Adjust fingers closing based on the item size (already set)
def adjust_fingers_closing_size(size_):

    full_opening = 0.175  # max distance finger to finger

    value_f = (full_opening - size_) * 100 / 2

    f = []
    f.append(value_f)
    f.append(value_f)
    f.append(value_f)

    return f


# Adjust fingers closing based on the diameter
def adjust_fingers_closing(coord):

    # Max
    full_opening = 0.175  # max distance finger to finger

    # Find item's diameter
    diameter = math.sqrt((coord[0]-coord[2])**2 + (coord[1]-coord[3])**2)
    value_f = (full_opening - diameter) * 100 / 2

    f = []
    f.append(value_f)
    f.append(value_f)
    f.append(value_f)

    return f
