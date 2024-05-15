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


![Face Detection](https://drive.google.com/uc?export=download&id=1fJjDlOX-1h0D5_P4NkgkRT6YbQV_3Oeh)


## 2. Face Recognition

Description: Exploring the application of Eigenfaces for face recognition, utilizing a dataset consisting of 5 individuals, each with 12 facial images. The dataset was divided into a training set, comprising 9 images per person, and a testing set, comprising 3 images per person:

- The core methodology involved applying Principal Component Analysis (PCA) to the training images to extract the most discriminative features. By computing the eigenvectors corresponding to the largest eigenvalues, we obtained a set of basis vectors known as eigenfaces, which capture the essential facial characteristics shared across the dataset.

- These eigenfaces formed a lower-dimensional space in which all facial images could be represented. Subsequently, we projected both the training and testing datasets onto this eigenface space. This transformation enabled us to effectively reduce the dimensionality of the data while preserving the essential facial information.

- During the recognition phase, new facial images were projected onto the same eigenface space. By measuring the similarity between the projected test image and the training dataset, we assigned the test image to the closest matching individual based on a predefined threshold or distance metric.

- Through this approach, we aimed to demonstrate the effectiveness of Eigenfaces in accurately recognizing faces across varying conditions such as lighting, facial expressions, and minor occlusions. The project contributes to the understanding and application of dimensionality reduction techniques in the field of computer vision, particularly in the domain of face recognition.


![Face Recognition](https://drive.google.com/uc?export=download&id=1fJjDlOX-1h0D5_P4NkgkRT6YbQV_3Oeh)


## 3. ROC Curve

Description: The Receiver Operating Characteristic (ROC) curve is a graphical representation used to assess the performance of a binary classifier system as its discrimination threshold is varied. It plots two parameters which are the true positive rate and the false positive rate:


![ROC Curve](https://drive.google.com/uc?export=download&id=1fJjDlOX-1h0D5_P4NkgkRT6YbQV_3Oeh)


## Contributors

We would like to thank the following individuals for their contributions to this project:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/nourhan-ahmedd">
        <img src="https://github.com/nourhan-ahmedd.png" width="100px" alt="@nourhan-ahmedd">
        <br>
        <sub><b>Nourhan Ahmed </b></sub>
      </a>
    </td>
  <tr>
    <td align="center">
      <a href="https://github.com/OmarEmad101">
        <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
        <br>
        <sub><b>Omar Emad</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Omarnbl">
        <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
        <br>
        <sub><b>Omar Nabil</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KhaledBadr07">
        <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
        <br>
        <sub><b>Khaled Badr</b></sub>
      </a>
    </td>
  </tr> 
  <!-- New Row -->
    <td align="center">
      <a href="https://github.com/hanaheshamm">
        <img src="https://github.com/hanaheshamm.png" width="100px" alt="@hanaheshamm">
        <br>
        <sub><b>Hana Hesham</b></sub>
      </a>
    </td>
  </tr>
</table>

