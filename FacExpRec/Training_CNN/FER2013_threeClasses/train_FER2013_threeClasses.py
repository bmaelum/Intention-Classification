
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
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
batch_size = 128

def generateModel():
    ## Model architecture
    model = tf.keras.models.Sequential()

    def ninthRev():
        # ------------ First revision ---------------------------------------------------------------------------------------------
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding="same", activation="relu"))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        # model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(2, activation="softmax"))
        # ---------------------------------------------------------------------------------------------------------------------------

    def eigthRev():
        # ------------ First revision ----------------------------------------------------------------------------------------------
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(3, activation="softmax"))
        # ---------------------------------------------------------------------------------------------------------------------------

    def seventhRev():
        # ------------ First revision ----------------------------------------------------------------------------------------------
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(2, activation="softmax"))
        # ---------------------------------------------------------------------------------------------------------------------------
    def sixthRev():
        # ------------------------ Sixth revision (try-and-fail) ----------------------------------------------
        # Layer 1
        #model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        #model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        # Layer 2
        #model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        #model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        # # Layer 3
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        #
        # # Layer 4
        # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        # model.add(tf.keras.layers.Activation('relu'))

        # # Layer 5
        # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        # model.add(tf.keras.layers.Activation('relu'))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=3))

        # # Layer 6
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.25))

        # Layer 7
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.25))

        # Layer 7
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.25))

        # Layer 8
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.Activation('softmax'))
    def fifthRev():
        # ------------------------ Fifth version (LamUong) -----------------------------------------------------
        epochs = 1200
        model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='valid', input_shape=(48,48,1)))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

        model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        model.add(tf.keras.layers.Conv2D(64, 3, 3))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

        model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        model.add(tf.keras.layers.Conv2D(128, 3, 3))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        model.add(tf.keras.layers.Conv2D(128, 3, 3))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))

        model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(2))

        model.add(tf.keras.layers.Activation('softmax'))
        # ----------------------------------------------------------------------------------------------------
    def fourthRev():
        ##  --------------------- Fourth revision -------------------------------------------------------------------------------------------------------------
        #Layer 1
        model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(5,5), strides=1, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 2
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        # Layer 3
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(5,5), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 4
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))

        # Layer 5
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 6
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))

        # Layer 7
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 8
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Output softmax layer
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.Activation('softmax'))

        ##-------------------------------------------------------------------------------------------------------------------------------------------------------------
    def thirdRev():
        #------------- Third revision (ImageNet/AlexNet) ---------------------------------
        #Layer 1
        #model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(3,3)))

        # Layer 2
        #model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))

        # Layer 3
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 4
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 5
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))

        # Layer 6
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 7
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 8
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.Activation('softmax'))
    def secondRev():
        # ---------------------------- Second revision ----------------------------
        # Must define the input shape in the first layer of the neural network
        model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(48,48,1)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.Activation('sigmoid'))
        # -------------------------------------------------------------------------
    def firstRev():
        # ------------ First revision ----------------------------------------------------------------------------------------------
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        # ---------------------------------------------------------------------------------------------------------------------------

    if sys.argv[2] == "firstRev":
        firstRev()

    elif sys.argv[2] == "secondRev":
        secondRev()

    elif sys.argv[2] == "thirdRev":
        thirdRev()

    elif sys.argv[2] == "fourthRev":
        fourthRev()

    elif sys.argv[2] == "fifthRev":
        fifthRev()

    elif sys.argv[2] == "sixthRev":
        sixthRev()

    elif sys.argv[2] == "seventhRev":
        seventhRev()

    elif sys.argv[2] == "eigthRev":
        eigthRev()

    elif sys.argv[2] == "ninthRev":
        ninthRev()



    #thirdRev()


    return model

def dataAugmentation():
    print "Data augmentation..."
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
            '/home/bjornar/MScDissertation/TrainingData/FER_data/AngryHappyNeutral/train',
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
            '/home/bjornar/MScDissertation/TrainingData/FER_data/AngryHappyNeutral/validation',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
            '/home/bjornar/MScDissertation/TrainingData/FER_data/AngryHappyNeutral/test',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    classDict = train_generator.class_indices
    print classDict

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
    # print old_stdout



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
    model.compile(  loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    # Load the weights with the best validation accuracy
    model.load_weights(wName)

    a, b, test_gen, c = dataAugmentation()


    # Evaluate the model on test set
    score = model.evaluate_generator(test_gen, 1372/batch_size)
    scoreStr = "Test accuracy = " + str(score[1])
    print scoreStr

    # Print test accuracy
    f = open(directoryName, "a")
    f.write(scoreStr)
    f.close()

    # y_hat = model.predict(x_test)

    #print "X_TEST TYPE: " + str(type(x_test))
    #print x_test[0]

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

    print imageToPred.shape
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
    print scoreStr

    # # Print test accuracy
    # f = open(directoryName, "a")
    # f.write(scoreStr)
    # f.close()


    y_hat = model.predict(imageToPred)
    imgClass = model.predict_classes(imageToPred)
    imgProb = model.predict_proba(imageToPred)
    print "Predicted class: " + str(imgClass)

    #print "X_TEST TYPE: " + str(type(x_test))
    #print x_test[0]

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
epochs = 2000
redirectionDirectory = "/home/bjornar/ML_models/FER/angryHappyNeutral"
foldername = "tf_keras_weights_" + sys.argv[2]
weightModName = redirectionDirectory + foldername + "/" + foldername + ".hdf5"
modelname = redirectionDirectory + foldername + "tf_keras_model_" + sys.argv[2] + ".hdf5"
print modelname
txtFile = redirectionDirectory + foldername+"/"+foldername+".txt"
if sys.argv[1] == 'train':
    print "----------------------------------------------------------------"
    if not os.path.exists(redirectionDirectory + foldername):
        os.makedirs(redirectionDirectory + foldername)

    if not os.path.exists(txtFile):
        print "File does not exist..."
        f = open(txtFile, "w+")

    else:
        f = open(txtFile, "a")
    f.write("\n\n----------------------------------------------------------------------------------------------- \n\n")
    f.close()

    if not os.path.exists(redirectionDirectory + foldername):
        os.makedirs(redirectionDirectory + foldername)


    print "Training model..."
    print "Running model with " + sys.argv[2]
    model = trainModel(epochs, modelname, weightModName, txtFile)#, xTrain, yTrain, xTest, yTest, xVal, yVal)
    print "Predicting images..."
    predict(redirectionDirectory + foldername + "/" + foldername + ".hdf5", txtFile)

    if not os.path.exists(txtFile):
        f = open(txtFile, "w+")
    else:
        f = open(txtFile, "a")
    f.write("\n\n----------------------------------------------------------------------------------------------- \n\n")
    f.close()

    print "----------------------------------------------------------------"
if sys.argv[1] == 'predict':
    print sys.argv[1]
    print sys.argv[2]

    predFromImage("smiling-makes-you-happy-1.jpg", weightModName)
