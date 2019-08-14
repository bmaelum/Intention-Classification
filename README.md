# Intention Classification 
## Master of Science Dissertation 
Project at the University of Bristol. **Dissertation can be found [here](https://www.dropbox.com/s/4h1am0xqd3xizbv/MSc_Robotics_BjornarMaelum.pdf?dl=0).**

This repo describes and demos how the computer vision project of Intention Classification works.

**Goal:**
Working project that can run on OS X. 

**Progress:**
- [x] Head Pose Estimation working on Ubuntu 16.04 and Python 2.7
- [ ] Head Pose Estimation working on Mac OS X and Python 3

## Abstract
The contribution of this research is a system enabling non-verbal human robot interaction
through the use of facial expression recognition and head pose estimation. The research is
looking into creating an interface for humans with quadriplegia or other similar conditions.
The interface is designed to enable non-verbal and non-physical interaction. The outcome of
this research is a system consisting of intention classification and robot arm manipulation. The
facial expression recognition and head pose estimation are mapped to an intention signaling
a task to be performed by the robot. The robot arm performs tasks such as pick-and-place of
an object and handing the object to the human. The facial expression recognition has been
trained using a convolutional neural network while the head pose estimation is based on facial
landmark detection through the use of randomized regression trees. The facial expression
recognition and head pose are used to represent a unique language for communication. The
robot arm used is an assistive robot arm performing tasks based on interaction through the
communication language. An experiment has been conducted gathering evidence that the
proposed system can provide reliable classification of intention within the defined language.
In the experiment, the human and robot interact using the developed system. The experiment
has been recorded and is presented in this document.

## Conceptual Design
<p align="center">
  <img width="400" src="https://github.com/bmaelum/Intention-Classification/blob/master/images/ConceptualDesign_v3.png">
</p>
