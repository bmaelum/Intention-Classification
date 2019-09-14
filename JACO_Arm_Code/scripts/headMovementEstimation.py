from collections import Counter

def mostCommon(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def calculateDirection(nosemarks):
    movementFactor = 15
    if len(nosemarks[9]) == 2:
        dX = nosemarks[9][0] - nosemarks[0][0]
        dY = nosemarks[9][1] - nosemarks[0][1]

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
