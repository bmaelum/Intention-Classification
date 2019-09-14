import cv2
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def detectFace(frame, faceCascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    return gray, faces

def cropFaces(gray, faces):
    face_crop = []
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
            # Factor makes sure we get a bigger picture of the face
            factor = 0.3
            h_factor = int(float(h * factor)) / 2
            w_factor = int(float(w * factor)) / 2
            face_crop.append(gray[y-h_factor:y+h+h_factor, x-w_factor:x+w+w_factor])

    return face_crop

def displayFace(fcs):
    for (x,y,w,h) in fcs:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("", frame)
    cv2.waitKey(0)

def displayCrop(img):
    # Display the cropped face
    for face in img:
        cv2.imshow("A cropped face", face)
        cv2.waitKey(0)

def predictImg(image):
    image = cv2.resize(image, (48,48))#, fx=1, fy=1, interpolation = cv2.INTER_CUBIC)

    #print "image.shape" + str(image.shape)
    cv2.imshow("image", image)

    image = np.array(image).astype(int)
    image = np.array(image).astype("float32")
    image = np.divide(image, 255)

    w, h = 48, 48
    image = image.reshape(1,48,48,1)

    # Define text labels
    exp_labels = [          "Angry",     # index 0
                            "Disgust",   # index 1
                            "Fear",      # index 2
                            "Happy",     # index 3
                            "Sad",       # index 4
                            "Surprise",  # index 5
                            "Neutral"]   # index 6

    model = tf.keras.models.load_model("tf_keras_model_10epochs.hdf5")
    classPred = model.predict_classes(image)
    probPred = model.predict_proba(image)
    probPred = probPred[0]


    pred = model.predict(image)
    for j in range(0,7):
        probPred[j] = probPred[j] * 100
        probPred[j] = round(probPred[j], 2)
        print str(exp_labels[j]) + ": " + str(probPred[j]) + "%"

    #print("Probability prediction => " + str(probPred))
    classPred = int(classPred)
    print("Class prediction => " + str(classPred) + "(" + exp_labels[classPred] +") " + str(probPred[classPred]) + "%")

    # Display image and prediction
    figure = plt.figure()
    plt.imshow(np.squeeze(image))
    plt.title("{}".format( exp_labels[classPred]))
    plt.show()
## --------------- MAIN ------------------

# This function detects a face and returns the cropped face(s)
cwd = os.getcwd()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
time.sleep(2.5)

#cv2.imshow(video_capture)

face_index = 0

#while True:
# Capture frame-by-frame
ret, frame = video_capture.read()

gray, faces = detectFace(frame, faceCascade)

face_crop = cropFaces(gray, faces)
#displayCrop(face_crop)

# if runWhile == 1:
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

video_capture.release()
cv2.destroyAllWindows()


testImg = face_crop
if len(testImg) == 1:
    print "Detected 1 face."
    for face in testImg:
        cv2.imwrite("croppedface.png", face)
        print face.shape
        if face.shape[0] == 0:
            print "Something is fishy"

        predictImg(face)

elif len(testImg) > 1:
    print "Detected " + str(len(testImg)) + " faces."
    for i in range(0, len(testImg)):
        predictImg(testImg[i])

else:
    print "Did not detect any faces."
