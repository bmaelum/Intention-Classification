
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tf.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import manipulateImg
import os
import csv
import cv2


# dimensions of our images.
img_width, img_height = 48, 48

#top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
epochs = 50
batch_size = 128


def loadImgData():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []

    numTrain = 25120
    numTest = 3589
    numVal = 3589
    for i in range(0, numTrain):
        image = cv2.imread("data/train/" + str(i) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("", image)
        x_train.append(image)

    for i in range(0, numTest):
        image = cv2.imread("data/test/" + str(i) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("", image)
        x_test.append(image)

    for i in range(0, numVal):
        image = cv2.imread("data/validation/" + str(i) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("", image)
        x_val.append(image)

    print len(x_train)
    print len(x_test)
    print len(x_val)
    # cv2.imshow("hei", x_train[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    file = open('/home/bjornar/TrainingData/fer2013/fer2013.csv')
    csv_f = csv.reader(file)

    for row in csv_f:
        if str(row[2]) == 'Training':# or str(row[2] == 'PublicTest'):
            y_train.append(row[0]) # labels
            #print row[2]

        elif str(row[2]) == 'PublicTest':# or str(row[2] == 'PublicTest'):
            y_train.append(row[0]) # labels
            #print row[2]

        elif str(row[2]) == 'PrivateTest':
            y_val.append(row[0])

    print "y_train: " + str(len(y_train))
    print "y_test: " + str(len(y_test))
    print "y_val: " + str(len(y_val))
    # print "Number of training images: " + str(len(x_train))
    # print "Number of validation images: " + str(len(x_val))

    return x_train, y_train, x_test, y_test, x_val, y_val

def toList(data):
    tempList = []
    for row in data:
        row = row.split(" ")
        #row = [ int(x) for x in row ]
        tempList.append(row)

    return tempList

def preprocessingImg():

    print("--- DATA PREPROCESSING ---")

    # x_train, y_train, x_test, y_test, x_val, y_val = loadImgData()
    # print("Size of dataset before preprocessing")
    # print "Train: " + str(len(x_train))
    # print "Test: " + str(len(x_test))
    # print "Validation: " + str(len(x_val))
    #
    # # print("Number of train data - " + str(len(x_train)))
    # # print("Number of validation data  - " + str(len(x_val)))
    #
    # # Define text labels
    # exp_labels = [          "Angry",     # index 0
    #                         "Disgust",   # index 1
    #                         "Fear",      # index 2
    #                         "Happy",     # index 3
    #                         "Sad",       # index 4
    #                         "Surprise",  # index 5
    #                         "Neutral"]   # index 6
    #
    #
    # ## Split data into train/validation/test datasets
    # # Break training data into train / validation sets (5k to validamtion)
    # # print("Rearranging train/test/validation data...")
    # # (x_train, x_test) = x_train[:28709], x_train[28709:] # pixels
    # # (y_train, y_test) = y_train[:28709], y_train[28709:] # labels
    #
    # # Data normalization
    # x_train     = np.asarray(x_train).astype("float32") / 255
    # x_test      = np.asarray(x_test).astype("float32") / 255
    # x_val       = np.asarray(x_val).astype("float32") / 255
    #
    # print x_train.shape
    # # Reshape images
    # x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    # print "x_train[0].shape: " + str(x_train[0].shape)
    #
    # # One-hot encode the labels
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_val = tf.keras.utils.to_categorical(y_val, 10)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    #
    # print len(y_train)
    # print len(y_test)
    # print len(y_val)
    #
    # return x_train, y_train, x_test, y_test, x_val, y_val, exp_labels

def prepro():

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

    print("Dataset size: " + str(len(x_train) + len(x_test) + len(x_val)))
    print("Number of train data - " + str(len(x_train)))
    print("Number of test data  - " + str(len(x_test)))
    print("Number of validation data  - " + str(len(x_val)))


    # print type(x_train)
    # print x_train.size
    # print x_train.shape

    ## Adding size to dataset by manipulating
    # Manipulating brightness
    #x_train_darker     = x_train
    #x_test_darker      = x_test
    #x_val_darker       = x_val
    # x_train_brighter   = x_train
    # x_test_brighter    = x_test
    # x_val_brighter     = x_val

    brigthnessFactor = 100

    #x_train_darker, x_test_darker, x_val_darker         = manipulateImg.darkenImg(x_train_darker, x_test_darker, x_val_darker, brigthnessFactor)
    #x_train_brighter, x_test_brighter, x_val_brighter   = brightenImg(x_train_brighter, x_test_brighter, x_val_brighter, brigthnessFactor)

    # Data normalization
    x_train     = x_train.astype("float32") / 255
    x_test      = x_test.astype("float32") / 255
    x_val       = x_val.astype("float32") / 255
    #x_train_darker =x_train_darker.astype("float32") / 255
    #x_test_darker = x_test_darker.astype("float32") / 255
    #x_val_darker = x_val_darker.astype("float32") / 255

    # print type(x_train_darker)
    # print x_train_darker.size
    # print x_train_darker.shape

    ## Append manipulated images to original dataset
    #print "Merging original and manipulated data..."
    #x_train  = np.append(x_train,   x_train_darker,     axis = 0)
    #x_train  = np.append(x_train,   x_train_brighter,   axis = 0)
    # y_train_orig = y_train
    #y_train  = np.append(y_train,   y_train, axis=0)
    #y_train  = np.append(y_train,   y_train_orig, axis=0)
    #print "Training dataset sizes:"
    #print x_train.shape
    #print y_train.shape

    #x_test   = np.append(x_test,    x_test_darker,      axis = 0)
    #x_test   = np.append(x_test,    x_test_brighter,    axis = 0)
    # y_test_orig = y_test
    #y_test   = np.append(y_test,    y_test, axis=0)
    #y_test   = np.append(y_test,    y_test_orig, axis=0)
    # print "Test dataset sizes:"
    # print x_test.shape
    # print y_test.shape

    #x_val    = np.append(x_val,     x_val_darker,       axis = 0)
    #x_val    = np.append(x_val,     x_val_brighter,     axis = 0)
    # y_val_orig = y_val
    #y_val    = np.append(y_val,     y_val,  axis=0)
    #y_val    = np.append(y_val,     y_val_orig,  axis=0)
    # print "Validation dataset sizes:"
    # print x_val.shape
    # print y_val.shape

    # Reshape input data from (28,28) to (28,28,1)
    w, h = 48, 48
    x_train = x_train.reshape(x_train.shape[0], w, h)#, 1)
    x_val   = x_val.reshape(x_val.shape[0], w, h, 1)
    x_test  = x_test.reshape(x_test.shape[0], w, h, 1)


    print "x_train[0].shape: " + str(x_train[0].shape)
    plt.imshow(x_train[0])
    plt.show()
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    print "x_train[0].shape: " + str(x_train[0].shape)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 7)
    y_val = tf.keras.utils.to_categorical(y_val, 7)
    y_test = tf.keras.utils.to_categorical(y_test, 7)

    # Print training set shape
    print("x_train shape: ", x_train.shape, "y_train shape: ", y_train.shape )

def trainModel(epochs, modelname): # x_train, y_train, x_test, y_test, x_val, y_val):

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
    ## Model architecture
    model = tf.keras.Sequential()

    ## ------------------------ Fifth version (LamUong) -----------------------------------------------------
    # epochs = 1200
    # model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='valid', input_shape=(48,48,1)))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
    #
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    # model.add(tf.keras.layers.Conv2D(64, 3, 3))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    #
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    # model.add(tf.keras.layers.Conv2D(128, 3, 3))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    # model.add(tf.keras.layers.Conv2D(128, 3, 3))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    #
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.PReLU(alpha_initializer='zeros'))
    # model.add(tf.keras.layers.Dropout(0.2))
    #
    # model.add(tf.keras.layers.Dense(7))
    #
    # model.add(tf.keras.layers.Activation('softmax'))
    ## ----------------------------------------------------------------------------------------------------
    ##  --------------------- Fourth revision -------------------------------------------------------------------------------------------------------------
    # Layer 1
    # model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(5,5), strides=1, padding='same', input_shape=(48,48,1), data_format="channels_last"))
    # model.add(tf.keras.layers.Activation('relu'))
    #
    # # Layer 2
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #
    # # Layer 3
    # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(5,5), padding='same', data_format="channels_last"))
    # model.add(tf.keras.layers.Activation('relu'))
    #
    # # Layer 4
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))
    #
    # # Layer 5
    # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
    # model.add(tf.keras.layers.Activation('relu'))
    #
    # # Layer 6
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))
    #
    # # Layer 7
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    # # Layer 8
    # model.add(tf.keras.layers.Dense(128))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    # # Output softmax layer
    # model.add(tf.keras.layers.Dense(7))
    # model.add(tf.keras.layers.Activation('softmax'))

    ##-------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ------------- Third revision (ImageNet/AlexNet) ---------------------------------
    compFactor = 4
    Layer 1
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
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=3))

    # Layer 3
    model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
    model.add(tf.keras.layers.Activation('relu'))

    # Layer 4
    model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
    model.add(tf.keras.layers.Activation('relu'))

    # Layer 5
    model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=3))

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
    model.add(tf.keras.layers.Dense(7))
    model.add(tf.keras.layers.Activation('softmax'))

    #--------------------------------------------------------------------------

    ## ---------------------------- Second revision ----------------------------
    # # Must define the input shape in the first layer of the neural network
    # model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(48,48,1)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #
    # model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(64))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(10))
    # model.add(tf.keras.layers.Activation('sigmoid'))
    ## -------------------------------------------------------------------------

    ## ------------ First revision ----------------------------------------------------------------------------------------------
    # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    # model.add(tf.keras.layers.Dropout(0.3))
    #
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    # model.add(tf.keras.layers.Dropout(0.3))
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(7, activation="softmax"))
    ## ---------------------------------------------------------------------------------------------------------------------------

    # Take a look at the model summary
    model.summary()

    ## Compile model
    print("\n--- COMPILE MODEL ---")
    model.compile(  loss="categorical_crossentropy",
                    optimizer="rmsprop",
                    metrics=["accuracy"])

    print "Data augmentation..."
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255)#,
        # rotation_range=50,
        # vertical_flip=True)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    classDict = train_generator.class_indices
    print classDict


    ## Train model
    print("\n--- TRAIN MODEL ---")
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
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

    tf.keras.models.save_model(model, modelname)

    return model

def predict(model):
    # Load the weights with the best validation accuracy
    tf.keras.model.load_weights("model.weights.best.hdf5")

    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)

    # Print test accuracy
    print("\n", "Test accuracy", score[1])

    y_hat = model.predict(x_test)

    #print "X_TEST TYPE: " + str(type(x_test))
    #print x_test[0]

    # Plot random sample of 10 test images, predicted labels and ground truth
    figure = plt.figure(figsize=(20,8))
    for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
        ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        # Display each image
        ax.imshow(np.squeeze(x_test[index]))
        predict_index = np.argmax(y_hat[index])
        true_index = np.argmax(y_test[index])
        # Set the title for each image
        ax.set_title("{} ({})".format(  exp_labels[predict_index],
                                        exp_labels[true_index]),
                                        color=("green" if predict_index == true_index else "red"))
        plt.show()

### ----------------- MAIN -------------------
#loadImgData()
#xTrain, yTrain, xTest, yTest, xVal, yVal, exp_labels = prepro()

# epochs = 200
modelname = "tf_keras_model_" + str(epochs) + "epochs.hdf5"
model = trainModel(epochs, modelname)#, xTrain, yTrain, xTest, yTest, xVal, yVal)
#
# model = tf.keras.models.load_model(modelname)
# model = 0
# for i in range(0,10):
#     predict(model)
