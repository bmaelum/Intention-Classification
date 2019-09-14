import cv2
import numpy as np

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

def detectAndCropColor(frame, faceCascade):
    face_crop = []

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
            # Factor makes sure we get a bigger picture of the face
            factor = 0.3
            h_factor = int(float(h * factor)) / 2
            w_factor = int(float(w * factor)) / 2
            face_crop.append(frame[y-h_factor:y+h+h_factor, x-w_factor:x+w+w_factor])

    return face_crop

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

def displayFace(fcs, frame):
    for (x,y,w,h) in fcs:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("", frame)

def displayCrop(img):
    # Display the cropped face
    for face in img:
        cv2.imshow("A cropped face", face)

def predictImg(image, model):
    #print "ExpRec: image.shape = " + str(image.shape)
    image = np.resize(image, (48,48))#, fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #print "image.shape" + str(image.shape)
    #cv2.imshow("image", image)

    image = np.array(image).astype(int)
    image = np.array(image).astype("float32")
    image = np.divide(image, 255)

    w, h = 48, 48
    image = image.reshape(1,w,h,1)

    # Define text labels
    exp_labels = [          "Angry",     # index 0
                            "Happy",   # index 1
                            "Neutral",
                            ]

    classPred = model.predict_classes(image, verbose=0)
    probPred = model.predict_proba(image, verbose=0)
    probPred = probPred[0]


    pred = model.predict(image)
    for j in range(0,1):
        probPred[j] = probPred[j] * 100
        probPred[j] = round(probPred[j], 2)
        #print str(exp_labels[j]) + ": " + str(probPred[j]) + "%"

    #print("Probability prediction => " + str(probPred))
    classPred = int(classPred)
    if classPred == 0:
        classPrediction = 'angry'
    elif classPred == 1:
        classPrediction = 'happy'
    elif classPred == 2:
        classPrediction = 'neutral'


    #print("Class prediction => " + str(classPred) + "(" + exp_labels[classPred] +") " + str(probPred[classPred]) + "%")

    # # Display image and prediction
    # figure = plt.figure()
    # plt.imshow(np.squeeze(image))
    # #plt.imshow(image)
    # plt.title("{}".format( exp_labels[classPred]))
    # plt.show()

    return classPrediction

## Here goes the Facial Expression Recognition
def facialExpressionRecogntion(frame, model, cascade):
    #print "Facial Expression Recognition..."
    gray, faces = detectFace(frame, cascade)

    prediction = []

    if len(faces) > 0:
        croppedFace = cropFaces(frame, faces)
        #displayCrop(croppedFace)
        for face in croppedFace:
            #cv2.imwrite("croppedface.png", face)
            #print face.shape
            prediction.append(predictImg(face, model))
        #displayFace(faces, color)
        #cv2.waitKey(0)
        for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        #displayFace(faces, color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, prediction[0], (20,70), font, 1, (0,255,0), 4)
        cv2.imshow("", frame)

    return str(prediction)
