## --- Created by Bjørnar Kjenaas Mælum ---
## For the thesis named "Non-Verbal Human-Robot Interaction using Facial Expressions and Head Pose"
## MSc Robotics at the University of Bristol and University of West of England

## Script for training a Convolutional Neural Network for the detection of Facial Expressions.

## This script is optimized to run on Mac OS X using Python 3.6

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tf.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import os
import csv
import cv2
import sys

# dimensions of our images.
img_width, img_height = 48, 48

#top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/Users/bjornar/Documents/Intention-Classification/Training_Facial_Expression_Recognition/TrainingData/FER_data/BinaryClassification/train'
validation_data_dir = '/Users/bjornar/Documents/Intention-Classification/Training_Facial_Expression_Recognition/TrainingData/FER_data/BinaryClassification/validation'
test_data_dir = '/Users/bjornar/Documents/Intention-Classification/Training_Facial_Expression_Recognition/TrainingData/FER_data/BinaryClassification/test'

batch_size = 128

def generateModel():
    ## Model architecture
    model = tf.keras.models.Sequential()

    # ------------ First revision ---------------------------------------------------------------------------------------------
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    #model.add(tf.keras.layers.Dense(128, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    # ---------------------------------------------------------------------------------------------------------------------------

    return model

def dataAugmentation():
    print("Data augmentation...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=50)
        # vertical_flip=True)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255,
          rotation_range=50)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255)#,
          #rotation_range=50)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            #'/home/bjornar/MScDissertation/TrainingData/FER_data/BinaryClassification/train',
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
            validation_data_dir,
            #'/home/bjornar/MScDissertation/TrainingData/FER_data/BinaryClassification/validation',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            #'/home/bjornar/MScDissertation/TrainingData/FER_data/BinaryClassification/test',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    classDict = train_generator.class_indices
    print(classDict)

    return train_generator, validation_generator, test_generator, classDict

# def myPrint(s):
#     with open(foldName+"/"+foldName + ".txt", "w+") as f:
#         print(s, file=f)

def trainModel(epochs, modelname, weightName, txtName): # x_train, y_train, x_test, y_test, x_val, y_val):

    # from keras import backend as K
    # K.set_image_data_format('channels_first')

    # Define text labels
    exp_labels = [          "Angry",     # index 0
                            "Disgust",   # index 1
                            "Fear",      # index 2
                            "Happy",     # index 3
                            "Sad",       # index 4
                            "Surprise",  # index 5
                            "Neutral"]   # index 6


    model = generateModel()

    # Take a look at the model summary
    # from contextlib import redirect_stdout
    # with open(foldName+"/"+foldName+".txt", "w") as f:
    #     with redirect_stdout(f):
    orig_stdout = sys.stdout
    sys.stdout = open(txtName, "a")
    model.summary()
    sys.stdout.close()
    sys.stdout = orig_stdout

    # from cStringIO import StringIO
    # old_stdout = sys.stdout
    # sys.stdout = model.summary = StringIO()
    # print(old_stdout



    ## Compile model
    print("\n--- COMPILE MODEL ---")
    model.compile(  loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])


    train_generator, validation_generator, test_generator, classDict = dataAugmentation()

    ## Train model
    print("\n--- TRAIN MODEL ---")
    #from keras.callbacks import ModelCheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=weightName, verbose = 1, save_best_only=True)
    # model.fit(   x_train,
    #              y_train,
    #              batch_size=128,
    #              epochs=epochs,
    #              validation_data=(x_val, y_val),
    #              callbacks=[checkpointer])
    model.fit_generator(
            train_generator,
            steps_per_epoch=250,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800,
            callbacks=[checkpointer])

    #tf.keras.utils.plot_model(model, to_file='model.png')

    tf.keras.models.save_model(model, modelname)

    return model

def predict(wName, directoryName):

    model = generateModel()

    ## Compile model
    print("\n--- COMPILE MODEL ---")
    model.compile(  loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    # Load the weights with the best validation accuracy
    model.load_weights(wName)

    a, b, test_gen, c = dataAugmentation()


    # Evaluate the model on test set
    score = model.evaluate_generator(test_gen, 1372/batch_size)
    scoreStr = "Test accuracy = " + str(score[1])
    print(scoreStr)

    # print(test accuracy
    f = open(directoryName, "a")
    f.write(scoreStr)
    f.close()

    # y_hat = model.predict(x_test)

    #print("X_TEST TYPE: " + str(type(x_test))
    #print(x_test[0]

    # # Plot random sample of 10 test images, predicted labels and ground truth
    # figure = plt.figure(figsize=(20,8))
    # for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    #     ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    #     # Display each image
    #     ax.imshow(np.squeeze(x_test[index]))
    #     predict_index = np.argmax(y_hat[index])
    #     true_index = np.argmax(y_test[index])
    #     # Set the title for each image
    #     ax.set_title("{} ({})".format(  exp_labels[predict_index],
    #                                     exp_labels[true_index]),
    #                                     color=("green" if predict_index == true_index else "red"))
    #     plt.show()

def predFromImage(imageToPred, modelNm):

    imageToPred = cv2.imread(imageToPred)
    imageToPred = cv2.cvtColor(imageToPred, cv2.COLOR_BGR2GRAY)
    imageToPred = cv2.resize(imageToPred, (48,48))
    imageToPred = np.reshape(imageToPred, (1,48,48,1))

    print(imageToPred.shape)
    model = generateModel()

    ## Compile model
    print("\n--- COMPILE MODEL ---")
    model.compile(  loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    # Load the weights with the best validation accuracy
    model.load_weights(modelNm)

    a, b, test_gen, c = dataAugmentation()


    # Evaluate the model on test set
    score = model.evaluate_generator(test_gen, 1372/batch_size)
    scoreStr = "Test accuracy = " + str(score[1])
    print(scoreStr)

    # # print(test accuracy
    # f = open(directoryName, "a")
    # f.write(scoreStr)
    # f.close()


    y_hat = model.predict(imageToPred)
    imgClass = model.predict_classes(imageToPred)
    imgProb = model.predict_proba(imageToPred)
    print("Predicted class: " + str(imgClass))

    #print("X_TEST TYPE: " + str(type(x_test))
    #print(x_test[0]

    # # Plot random sample of 10 test images, predicted labels and ground truth
    # figure = plt.figure(figsize=(20,8))
    # for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    #     ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    #     # Display each image
    #     ax.imshow(np.squeeze(x_test[index]))
    #     predict_index = np.argmax(y_hat[index])
    #     true_index = np.argmax(y_test[index])
    #     # Set the title for each image
    #     ax.set_title("{} ({})".format(  exp_labels[predict_index],
    #                                     exp_labels[true_index]),
    #                                     color=("green" if predict_index == true_index else "red"))
### ----------------- MAIN -------------------
#epochs = 2000
try:
    if sys.argv[3]:
        epochs = int(sys.argv[3])
except:
    epochs = 2000
print("Epochs: " + str(epochs))

redirectionDirectory = "/Users/bjornar/ML_models/FER/"
foldername = "tf_keras_weights"
weightModName = redirectionDirectory + foldername + "/" + foldername + ".hdf5"
modelname = redirectionDirectory + foldername + "tf_keras_model.hdf5"
print(modelname)
txtFile = redirectionDirectory + foldername+"/"+foldername+".txt"
if sys.argv[1] == 'train':
    print("----------------------------------------------------------------")
    if not os.path.exists(redirectionDirectory + foldername):
        os.makedirs(redirectionDirectory + foldername)

    if not os.path.exists(txtFile):
        print("File does not exist...")
        f = open(txtFile, "w+")

    else:
        f = open(txtFile, "a")
    f.write("\n\n----------------------------------------------------------------------------------------------- \n\n")
    f.close()

    if not os.path.exists(redirectionDirectory + foldername):
        os.makedirs(redirectionDirectory + foldername)


    print("Training model...")
    #print("Running model with " + sys.argv[2])
    model = trainModel(epochs, modelname, weightModName, txtFile)#, xTrain, yTrain, xTest, yTest, xVal, yVal)
    print("Predicting images...")
    predict(redirectionDirectory + foldername + "/" + foldername + ".hdf5", txtFile)

    if not os.path.exists(txtFile):
        f = open(txtFile, "w+")
    else:
        f = open(txtFile, "a")
    f.write("\n\n----------------------------------------------------------------------------------------------- \n\n")
    f.close()

    print("----------------------------------------------------------------")
if sys.argv[1] == 'predict':
    print(sys.argv[1])
    #print(sys.argv[2])

    print("Predict image with ground truth happy")
    predFromImage("happy_person.png", weightModName)

    print("\n\n-- Predict image with ground truth angry --")
    predFromImage("angry_person.png", weightModName)
