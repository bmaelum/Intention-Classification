"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from multiprocessing import Process, Queue

import numpy as np
import cv2
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import time
from collections import Counter

def mostCommon(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
# multiprocessing may not work on Windows and macOS, check OS for safety.
#detect_os()

CNN_INPUT_SIZE = 128


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        print "HERE"
        print image.shape
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def calculateDirection(nosemarks):
    # print 'Calculating direction...'
    # print nosemarks
    # print nosemarks[9]
    movementFactor = 15
    # print nosemarks[0]
    if len(nosemarks[9]) == 2:
        dX = nosemarks[9][0] - nosemarks[0][0]
        dY = nosemarks[9][1] - nosemarks[0][1]
        # print"dX = " + str(dX)
        # print "dY = " + str(dY)
        # if dX < -movementFactor:
        #     dXdirection = "east"
        # elif dX > movementFactor:
        #     dXdirection = "west"
        # else:
        #     dXdirection = "No direction detected."
        #
        # if dY > movementFactor:
        #     dYdirection = "north"
        # elif dY < -movementFactor:
        #     dYdirection = "south"
        # else:
        #     dYdirection = "No direction detected."

        if dX < -movementFactor and dY < -movementFactor:
            direction = 'south east'
        elif dX < -movementFactor and dY > movementFactor:
            direction = 'north east'
        elif dX > movementFactor and dY < -movementFactor:
            direction = 'south west'
        elif dX > movementFactor and dY > movementFactor:
            direction = 'north west'
        elif dX < -movementFactor and abs(dY) < movementFactor:
            direction = 'east'
        elif dX > movementFactor and abs(dY) < movementFactor:
            direction = 'west'
        elif abs(dX) < movementFactor and dY < -movementFactor:
            direction = 'south'
        elif abs(dX) < movementFactor and dY > movementFactor:
            direction = 'north'

        else:
            direction = 'still'

    return direction

def headPoseEstimation():
    print "HEAD POSE ESTIMATION..."

    """MAIN"""
    # Video source from webcam or video file.
    video_src = 0
    #video_src = 'EWSN.avi'
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    print height
    print width
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    noseMarks = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    numXPoints = 0
    numYPoints = 0
    sumX = 0
    sumY = 0

    # start = time.time()
    # previousTime1 = start
    # previousTime2 = start
    currentTime = 0
    previousTime1 = 0
    previousTime2 = 0

    directionArray = []
    moveSequence = []
    moves = []
    classifyMoves = 0
    headPoseDirection = 'emtpy'

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        print frame.shape

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()
        #print type(facebox)
        #print facebox

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            print face_img.shape
            marks = mark_detector.detect_marks(face_img)

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # # Get average position of nose
            noseMarksTemp = []
            noseMarksTemp.append(marks[30][0])
            noseMarksTemp.append(marks[30][1])
            noseMarks[0] = noseMarksTemp

            for i in range(9, 0, -1):
                noseMarks[i] = noseMarks[i-1]

            # Get the direction of head movement
            headPoseDirection = calculateDirection(noseMarks)
            #if headPoseDirection != 'still':
            directionArray.append(headPoseDirection)
            #print directionArray
            #print len(directionArray)
            print "To capture a movement press 'a' and perform a movement."
            #currentTime1 = time.time()
            if cv2.waitKey(5) == ord('a') and not classifyMoves:
                classifyMoves = 1
                print "Start classifying movement..."
                timer = time.time()
                currentTime = timer
                previousTime1 = timer
                previousTime2 = timer

            if cv2.waitKey(5) == ord('b') and classifyMoves:
                classifyMoves = 0
                print "Stopped classifying movement..."

            currentTime = time.time()
            if currentTime - previousTime1 > 2 and classifyMoves:
                print "------------------------------------------------"
                print "Elapsed timer 1: " + str(currentTime - previousTime1)
                #print len(directionArray)

                # Get most common direction
                moveClass = mostCommon(directionArray)
                #moveSequence.append(moveClass)
                print moveClass

                # Get a sequence of head movements
                # if currentTime - previousTime2 > 10 and classifyMoves == 1 and len(moves) == 0:
                #     print "Elapsed timer 2: " + str(currentTime - previousTime2)
                #     numMoves = len(moveSequence)
                #     moves = moveSequence[(numMoves-5):(numMoves-1)]
                #     print moves
                #     moveSequence = []
                #     previousTime2 = currentTime
                #     classifyMoves = 0
                classifyMoves = 0
                directionArray = []
                previousTime1 = currentTime

                print "To continue type 'c' or to recapture a movement type 'a'."

            if cv2.waitKey(5) == ord('c'):
                break


            #print previousTime
            # Uncomment following line to show raw marks.
            #mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            stabile_pose = np.reshape(stabile_pose, (-1, 3))

            # Uncomment following line to draw pose annotaion on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotaion on frame.
            #pose_estimator.draw_annotation_box(
            #    frame, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))

        # if len(moves) > 1:
        #     cv2.putText(frame, moves[0], (450,70), font, 1, (0,0,0), 4)
        #     cv2.putText(frame, moves[1], (450,100), font, 1, (0,0,0), 4)
        #     cv2.putText(frame, moves[2], (450,130), font, 1, (0,0,0), 4)
        #     cv2.putText(frame, moves[3], (450,160), font, 1, (0,0,0), 4)


        cv2.putText(frame, headPoseDirection, (20,70), font, 1, (0,255,0), 4)

        # Show preview.
        #cv2.namedWindow("", cv2.WINDOW_NORMAL)
        cv2.imshow("Preview", frame)
        #cv2.resizeWindow("preview", 5000,5000)
        if cv2.waitKey(5) == 27:
            break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

    return moveClass

headPoseEstimation()
