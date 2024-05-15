# Face_Detection_and_Face_recognition

## Table of Contents

1. [Face Detection](#face-detection)
2. [Face Recognition](#face-recognition)
3. [ROC Curve](#roc-curve)

## 1. Face Detection

Description: The face detection logic implemented in our system relies on the Haar cascade classifier provided by the OpenCV library. Here's an overview of the key aspects of this library and its functionality:
- Haar Cascade Classifier:
The Haar cascade classifier is a machine learning-based algorithm used for object detection, including face detection. It operates by analyzing the intensity patterns of pixels in an image to identify regions containing objects of interest.
- detectMultiScale Function:
This function is a fundamental component of the face detection process. It applies the Haar cascade classifier to the input image, scanning it at multiple scales to detect potential face regions. The function returns a list of rectangles representing the detected faces' positions and sizes.
- Parameters:
Several parameters, such as scaleFactor, minNeighbors, minSize, and maxSize, are provided to fine-tune the face detection process. These parameters control the sensitivity, accuracy, and computational efficiency of the detection algorithm.


![Filtering and noise](https://drive.google.com/uc?export=download&id=1-YYlvcxqidRxzZF_ObBdyYf8JpUlHilk)


## 2. Face Recognition

Description: The face detection logic implemented in our system relies on the Haar cascade classifier provided by the OpenCV library. Here's an overview of the key aspects of this library and its functionality:
- Haar Cascade Classifier:
The Haar cascade classifier is a machine learning-based algorithm used for object detection, including face detection. It operates by analyzing the intensity patterns of pixels in an image to identify regions containing objects of interest.
- detectMultiScale Function:
This function is a fundamental component of the face detection process. It applies the Haar cascade classifier to the input image, scanning it at multiple scales to detect potential face regions. The function returns a list of rectangles representing the detected faces' positions and sizes.
- Parameters:
Several parameters, such as scaleFactor, minNeighbors, minSize, and maxSize, are provided to fine-tune the face detection process. These parameters control the sensitivity, accuracy, and computational efficiency of the detection algorithm.
