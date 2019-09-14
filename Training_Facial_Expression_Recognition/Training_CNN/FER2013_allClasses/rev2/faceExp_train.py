
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tf.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import os
import csv
import cv2
import sys
*-# dimensions of our images.
img_width, img_height = 48, 48

#top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
batch_size = 128

def generateModel():
    ## Model architecture
    model = tf.keras.models.Sequential()
    def sixthRev():
        # ------------------------ Sixth revision (try-and-fail) ----------------------------------------------
        # Layer 1
        #model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same', input_shape=(48,48,1), data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        # Layer 2
        #model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        # # Layer 3
        # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format="channels_last"))
        # model.add(tf.keras.layers.Activation('relu'))
        #
        # # Layer 4
        # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        # model.add(tf.keras.layers.Activation('relu'))

        # # Layer 5
        # model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), padding='same', data_format='channels_last'))
        # model.add(tf.keras.layers.Activation('relu'))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=3))

        # # Layer 6
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(256))
        # model.add(tf.keras.layers.Activation('relu'))
        # model.add(tf.keras.layers.Dropout(0.5))

        # Layer 7
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 8
        model.add(tf.keras.layers.Dense(7))
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

        model.add(tf.keras.layers.Dense(7))

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
        model.add(tf.keras.layers.Dense(7))
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
        model.add(tf.keras.layers.Dense(7))
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
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Activation('sigmoid'))
        # -------------------------------------------------------------------------
    def firstRev():
        # ------------ First revision ----------------------------------------------------------------------------------------------
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(48,48,1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(7, activation="softmax"))
        # ---------------------------------------------------------------------------------------------------------------------------

    #firstRev()
    thirdRev()
    # Take a look at the model summary
    model.summary()

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

    train_generator = train_datagen.flow_from_directory(
            '/home/bjornar/MScDissertation/TrainingData/FER_data/train',
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
            '/home/bjornar/MScDissertation/TrainingData/FER_data/validation',#,
            color_mode='grayscale',
            target_size=(48, 48))
            # batch_size=32,
            # class_mode='categorical')#,
            #class_mode='binary')

    classDict = train_generator.class_indices
    print classDict

    return train_generator, validation_generator, classDict

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


    model = generateModel()

    ## Compile model
    print("\n--- COMPILE MODEL ---")
    model.compile(  loss="categorical_crossentropy",
                    optimizer="Adagrad",
                    metrics=["accuracy"])


    train_generator, validation_generator, classDict = dataAugmentation()

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

def predict():

    model = generateModel()
    # Load the weights with the best validation accuracy
    model.load_weights("model.weights.best.hdf5")

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
epochs = 2000
if sys.argv[1] == 'train':
    print "Training model..."
    modelname = "tf_keras_model_" + str(epochs) + "epochs.hdf5"
    model = trainModel(epochs, modelname)#, xTrain, yTrain, xTest, yTest, xVal, yVal)

if sys.argv[1] == 'predict':
    print "Predicting images..."
    predict()
