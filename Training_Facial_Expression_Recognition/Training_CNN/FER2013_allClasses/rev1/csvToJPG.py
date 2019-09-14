import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tf.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import manipulateImg

import cv2

import csv

def loadData():
    images = []
    labels = []

    file = open('/home/bjornar/TrainingData/fer2013/fer2013.csv')
    csv_f = csv.reader(file)
    # csvForCount = csv.reader(file)
    # row_count = sum(1 for row in csvForCount)  # fileObject is your csv.reader
    # print "Number of rows: " + str(row_count)
    num = 0
    #print csv_f.line_num
    for row in csv_f:
        if str(row[2]) == 'Training' or str(row[2]) == 'PublicTest' or str(row[2]) == 'PrivateTest':
            images.append(row[1]) # pixels
            labels.append(row[0]) # labels
        num += 1

    datasetSize = num
            #print row[2]

        # if str(row[2]) == 'PublicTest':# or str(row[2] == 'PublicTest'):
        #     x_train.append(row[1]) # pixels
        #     y_train.append(row[0]) # labels
        #     #print row[2]
        #
        # elif str(row[2]) == 'PrivateTest':
        #     x_val.append(row[1])
        #     y_val.append(row[0])

    # print "Number of training images: " + str(len(x_train))
    # print "Number of validation images: " + str(len(x_val))

    return images, labels, datasetSize

def toList(data):
    tempList = []
    for row in data:
        row = row.split(" ")
        #row = [ int(x) for x in row ]
        tempList.append(row)

    return tempList

def csvToImg():
    print("--- DATA PREPROCESSING ---")

    imgs, lbls, numSamples = loadData()
    print("Size of dataset: " + str(numSamples)) + " before preprocessing."

    # print("Number of train data - " + str(len(x_train)))
    # print("Number of validation data  - " + str(len(x_val)))

    # Define text labels
    exp_labels = [          "Angry",     # index 0
                            "Disgust",   # index 1
                            "Fear",      # index 2
                            "Happy",     # index 3
                            "Sad",       # index 4
                            "Surprise",  # index 5
                            "Neutral"]   # index 6


    ## Split data into train/validation/test datasets
    # Break training data into train / validation sets (5k to validamtion)
    print("Rearranging train/test/validation data...")
    (x_train, x_val, x_test) = imgs[:28709], imgs[28709:32298], imgs[32298:] # pixels
    (y_train, y_val, y_test) = lbls[:28709], lbls[28709:32298], lbls[32298:] # labels

    print len(x_train)
    print len(x_val)
    print len(x_test)
    print len(y_train)
    print len(y_val)
    print len(y_test)

    print type(x_train)
    #print x_train[0]

    print("Fixing dataformat...")
    print("1/6...")
    x_train = np.array(toList(x_train)).astype(int)
    print "x_train shape: " + str(x_train.shape)
    print("2/6...")
    y_train = np.array(toList(y_train)).astype(int)
    print("3/6...")
    x_test  = np.array(toList(x_test)).astype(int)
    print("4/6...")
    y_test  = np.array(toList(y_test)).astype(int)
    print("5/6...")
    x_val   = np.array(toList(x_val)).astype(int)
    print("6/6...")
    y_val   = np.array(toList(y_val)).astype(int)

    #x_train = tf.keras.utils.to_categorical(x_train)
    print "x_train to categorical: " + str(x_train.shape)


    print("Number of train data - " + str(len(x_train)))
    print("Number of test data  - " + str(len(x_test)))
    print("Number of validation data  - " + str(len(x_val)))

    # Reshape input data from (28,28) to (28,28,1)
    w, h = 48, 48
    x_train = x_train.reshape(x_train.shape[0], w, h)#, 1)
    x_val   = x_val.reshape(x_val.shape[0], w, h)#,, 1)
    x_test  = x_test.reshape(x_test.shape[0], w, h)#, 1)

    # size_x_train = x_train.shape[0]
    # print "x_train[0].shape: " + str(size_x_train)

    def writeImg(array, label, folder):
        print "Writing images to " + folder + "..."
        size = len(array)
        for i in range(0, size):
            if label[i] == 0:
                filename = folder + "/angry/" + str(i) + ".png"
            elif label[i] == 1:
                filename = folder + "/disgust/" + str(i) + ".png"
            elif label[i] == 2:
                filename = folder + "/fear/" + str(i) + ".png"
            elif label[i] == 3:
                filename = folder + "/happy/" + str(i) + ".png"
            elif label[i] == 4:
                filename = folder + "/sad/" + str(i) + ".png"
            elif label[i] == 5:
                filename = folder + "/surprise/" + str(i) + ".png"
            elif label[i] == 6:
                filename = folder + "/neutral/" + str(i) + ".png"

            else:
                print "Could not identify label!"
                break

            cv2.imwrite(filename, array[i])

    writeImg(x_train, y_train, 'data/train')
    writeImg(x_test, y_test, 'data/test')
    writeImg(x_val, y_val, 'data/validation')


    # Print training set shape
    print("x_train shape: ", x_train.shape, "y_train shape: ", y_train.shape )

    return x_train, y_train, x_test, y_test, x_val, y_val, exp_labels



csvToImg()
