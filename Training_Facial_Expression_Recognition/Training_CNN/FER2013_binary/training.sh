#!/bin/sh
python train_FER2013_binary.py train firstRev
python train_FER2013_binary.py train secondRev
python train_FER2013_binary.py train thirdRev
python train_FER2013_binary.py train fourthRev
python /home/bjornar/MScDissertation/FacExpRec/FER2013_train_and_predict/CNN_FaceExp/binaryClassification/train_FER2013_binary.py train fifthRev
python /home/bjornar/MScDissertation/FacExpRec/FER2013_train_and_predict/CNN_FaceExp/binaryClassification/train_FER2013_binary.py train sixthRev
python train_FER2013_binary.py train seventhRev
python train_FER2013_binary.py train eigthRev
