import cv2
import pyqtgraph
import pandas as pd
import imageio
import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import numpy as np
import os
from os import path
import sys
import numpy as np
import pyqtgraph as pg
from PIL import Image
import matplotlib.pyplot as plt
import skimage.exposure as exposure

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))


class ImageConverter:
    @staticmethod
    def numpy_to_pixmap(array):
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)

        # Check if the array is 2D or 3D
        if len(array.shape) == 2:
            # For 2D arrays (grayscale)
            height, width = array.shape
            bytes_per_line = width
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(array.shape) == 3 and array.shape[2] == 3:
            # For 3D arrays (RGB color images)
            height, width, channels = array.shape
            bytes_per_line = width * channels
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            print("Unsupported array shape.")
            return None

        # Convert the QImage to a QPixmap and return it
        return QPixmap.fromImage(img)


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.imagePath = None
        self.initializeUI()
        self.Handle_Buttons()
        pyqtgraph.setConfigOptions(antialias=True)
        # folder_path = r'C:\Users\zorof\Downloads\merged_train_data'
        folder_path = r'C:\PyCharm codes for hanona\Third Biomedical Year, Second Term\Computer Vision\Task 5 CV\Trial 4\merged_train_data'
        self.apply_pca(folder_path)
        self.get_refrences()
        self.threshold_slider.setMinimum(1000)
        self.threshold_slider.setMaximum(5000)
        self.threshold_slider.setValue(1500)  # Set initial value
        self.threshold_slider.setSingleStep(500)

    def initializeUI(self):
        self.setWindowTitle("Face Detetction and Face Recognition")

    def Handle_Buttons(self):
        self.actionUpload_Image.triggered.connect(lambda: self.openImageDialog(mode='1'))
        # self.actionUpload_Second_Image.triggered.connect(lambda: self.openImageDialog(mode='2'))
        self.face_detection_btn.clicked.connect(self.toggle_tab)
        self.face_recognition_btn.clicked.connect(self.toggle_tab)
        self.ROC_btn.clicked.connect(self.toggle_tab)
        self.detection_submit_btn.clicked.connect(self.detect_faces)
        self.pca_btn.clicked.connect(self.pca_btn_pressed)
        self.show_roc_button.clicked.connect(self.plot_roc)

    def openImageDialog(self, mode):
        # Open a file dialog to select an image
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if mode == '1':
            if imagePath:
                self.imagePath = imagePath
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)  # kk
                # Convert the image from BGR to RGB
                IMG = cv2.imread(self.imagePath)
                # Convert the image from BGR to RGB
                self.img_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
                self.show_image(QImage(imagePath), mode)
                # self.filtered_image.clear()
        else:
            if imagePath:
                # image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
                self.show_image(QImage(imagePath), mode)

    def show_image(self, image, mode):
        # Convert the QImage to a QPixmap once, outside the loop
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Create a list of all QLabel widgets that should display the image
        if mode == '1':
            labels = [
                self.original_image,
                self.face_recognition_image
            ]
            # Iterate over the list and set the pixmap on each QLabel
            for label in labels:
                scaled_pixmap = pixmap
                label.setPixmap(scaled_pixmap)
                label.setScaledContents(False)

        # if mode == '2':
        #     self.second_original_image_tab7.setPixmap(pixmap)
        #     self.second_original_image_tab7.setScaledContents(False)

    def toggle_tab(self):
        # Get the sender object to determine which button was clicked
        sender = self.sender()
        if sender == self.face_detection_btn:
            # Switch to the first tab of the stacked widget
            self.stackedWidget.setCurrentIndex(0)
        elif sender == self.face_recognition_btn:
            # Switch to the second tab of the stacked widget
            self.stackedWidget.setCurrentIndex(1)
        elif sender == self.ROC_btn:
            # Switch to the third tab of the stacked widget
            self.stackedWidget.setCurrentIndex(2)

    def detect_faces(self):

        # Load the pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(self.img_rgb, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                              maxSize=(200, 200))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(self.img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with detected faces
        resulted_pixmap = ImageConverter.numpy_to_pixmap(self.img_rgb)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.face_detection_image.setPixmap(scaled_pixmap)
        self.face_detection_image.setScaledContents(False)

    def apply_pca(self, path):
        data_matrix = self.train_images(path)
        mean_image = self.calculate_mean_image(data_matrix)
        self.mean_image = mean_image
        matrix_subtracted = self.subtract_mean_from_images(data_matrix, mean_image)
        cov_matrix = self.calculate_covariance_matrix(matrix_subtracted)
        eigenvalues, eigenvectors = self.calculate_normalized_eigenvectors(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        sorted_eigen_value, sorted_eigen_vectors = self.sort_eigenvectors(eigenvalues, eigenvectors)
        best_eigenvectors = sorted_eigen_vectors[:, :40]
        self.best_eigenvectors = best_eigenvectors
        data_reduced = np.dot(best_eigenvectors.T, matrix_subtracted)
        self.data_reduced = data_reduced
        # test_folder_path = r'C:\Users\zorof\Downloads\merged_test_data'

    def pca_btn_pressed(self):
        test_matrix = self.test_data()
        new_image_subtracted = self.subtract_mean_from_images(test_matrix, self.mean_image)
        # 4. Project onto Eigen Vectors
        new_image_projected = np.dot(self.best_eigenvectors.T, new_image_subtracted)
        distances = self.get_distances(new_image_projected, self.data_reduced)
        ids = []
        for i in range(new_image_projected.shape[1]):
            ids.append(self.get_id(distances[i]))
        # Convert the list of IDs to a string
        ids_str = ', '.join(ids)
        if ids_str == "Unknown":
            self.show_ref(self.ref_images[5])
        # Construct the final text to be displayed
        text = "ID: " + ids_str
        # Update the text of the id_label
        self.id_label.setText(text)

    def train_images(self, folder_path):
        # List all image files in the folder
        image_files = [file for file in os.listdir(folder_path) if
                       file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        images = []
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            # Load the image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Flatten image into a 1D vector
                image_vector = image.flatten()
                images.append(image_vector)
            else:
                print(f"Unable to load image: {image_path}")

        if images:
            # Construct the data matrix
            data_matrix = np.vstack(images)
            data_matrix = data_matrix.T
            print(data_matrix.shape)
            return data_matrix
        else:
            print("No images found in the folder or all images failed to load.")
            return None

    def calculate_mean_image(self, data_matrix):
        # Calculate the mean along the columns (axis 1) of the data matrix
        mean_image = np.mean(data_matrix, axis=1)
        print(mean_image.shape)
        return mean_image

    def subtract_mean_from_images(self, data_matrix, mean_image):
        # Subtract the mean image from all images
        data_matrix_subtracted = data_matrix - mean_image[:, np.newaxis]
        print(data_matrix_subtracted.shape)
        return data_matrix_subtracted

    def calculate_covariance_matrix(self, data_matrix_subtracted):
        # Calculate the covariance matrix
        covariance_matrix = np.cov(data_matrix_subtracted)
        print('cov', covariance_matrix.shape)
        return covariance_matrix

    def calculate_normalized_eigenvectors(self, covariance_matrix):
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        print('val', eigenvalues.shape)
        print('vec', eigenvectors.shape)
        return eigenvalues, eigenvectors

    def sort_eigenvectors(self, eigenvalues, eigenvectors):
        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalue = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        return sorted_eigenvalue, sorted_eigenvectors

    def test_data(self):
        test_images = []
        image_vector = self.image.flatten()
        test_images.append(image_vector)
        test_matrix = np.vstack(test_images)
        test_matrix = test_matrix.T
        return test_matrix

    def get_distances(self, new_image_projected, data_reduced):
        distances = []
        for i in range(new_image_projected.shape[1]):
            distance = np.linalg.norm(data_reduced - new_image_projected[:, i].reshape(-1, 1), axis=0)
            distances.append(distance)
        return (distances)

    def get_id(self, distances):
        threshold = 3000
        for index, distance in enumerate(distances):
            if (distance <= threshold):
                index = index // 8
                self.show_ref(self.ref_images[index])
                index += 1  # Adjust index to match your naming convention
                # Mapping index to names using dictionary
                names = {1: "Emad", 2: "Nabil", 3: "Khaled", 4: "Mohab", 5: "Zaza"}
                return names.get(index, "Unknown")
        return "Unknown"

    def get_refrences(self):
        # ref_folder_path = r'C:\Users\zorof\Downloads\refernces'
        ref_folder_path = r'C:\PyCharm codes for hanona\Third Biomedical Year, Second Term\Computer Vision\Task 5 CV\Trial 4\ref'
        ref_image_files = [file for file in os.listdir(ref_folder_path) if
                           file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        self.ref_images = []
        for image_file in ref_image_files:
            image_path = os.path.join(ref_folder_path, image_file)
            # Load the image in grayscale
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is not None:
                self.ref_images.append(image)
            else:
                print(f"Unable to load image: {image_path}")

    def show_ref(self, image):
        # Convert the ndarray to a QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Scale the pixmap to fit the QLabel
        scaled_pixmap = pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Display the pixmap in the QLabel
        self.reference.setPixmap(scaled_pixmap)
        self.reference.setScaledContents(False)

    def test_roc(self):
        self.roc_folder_path = r'C:\PyCharm codes for hanona\Third Biomedical Year, Second Term\Computer Vision\Task 5 CV\Trial 4\merged_test_data'
        self.roc_image_files = [file for file in os.listdir(self.roc_folder_path) if
                                file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        self.roc_images = []
        for image_file in self.roc_image_files:
            image_path = os.path.join(self.roc_folder_path, image_file)
            # Load the image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Flatten image into a 1D vector
                image_vector = image.flatten()
                self.roc_images.append(image_vector)
            else:
                print(f"Unable to load image: {image_path}")
        self.roc_matrix = np.vstack(self.roc_images)
        self.roc_matrix = self.roc_matrix.T
        self.roc_image_subtracted = self.subtract_mean_from_images(self.roc_matrix, self.mean_image)
        self.roc_image_projected = np.dot(self.best_eigenvectors.T, self.roc_image_subtracted)
        self.roc_distances = self.get_distances(self.roc_image_projected, self.data_reduced)
        # self.get_id()

    def get_id_roc(self, distances, threshold):
        # threshold=3000
        for index, distance in enumerate(distances):
            if (distance <= threshold):
                index = index // 8
                return index + 1
        return -1

    def plot_roc(self):
        self.test_roc()
        input_threshold = self.threshold_slider.value()
        dialog = QDialog(self)
        dialog.setWindowTitle("ROC Curve")
        dialog.resize(800, 600)  # Adjust the size as necessary

        # Create a figure and canvas
        figure = Figure()
        canvas = FigureCanvas(figure)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        dialog.setLayout(layout)

        # Plotting the ROC curve
        ax = figure.add_subplot(111)

        true_labels = np.array([-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, -1])
        # predicted_ids = [1, 1, 2, 1, 2, 2, 3, 2, 2, 3, 4, 3, 3, 5, 4, -1]  # Example prediction IDs
        # predicted_ids = [-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, -1]  # Example prediction IDs
        ids = []
        for i in range(self.roc_image_projected.shape[1]):
            ids.append(self.get_id_roc(self.roc_distances[i], threshold=input_threshold))
        print(ids)
        predicted_ids = ids

        # Determine the list of classes
        classes = np.unique(true_labels)

        # Store individual ROC curves
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        # Initialize accumulators for metrics
        acc_scores = []
        prec_scores = []
        spec_scores = []

        for cls in classes:
            # Prepare binary classification problem
            true_binary = (true_labels == cls).astype(int)
            pred_binary = (predicted_ids == cls).astype(int)

            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(true_binary, pred_binary)
            roc_auc = auc(fpr, tpr)

            # Store ROC components
            fpr_dict[cls] = fpr
            tpr_dict[cls] = tpr
            roc_auc_dict[cls] = roc_auc

            # Calculate accuracy, precision, and specificity
            acc = accuracy_score(true_binary, pred_binary)
            precision = precision_score(true_binary, pred_binary, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Accumulate scores
            acc_scores.append(acc)
            prec_scores.append(precision)
            spec_scores.append(specificity)

        # Average ROC Curve
        all_fpr = np.unique(np.concatenate([fpr_dict[cls] for cls in classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for cls in classes:
            mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
        mean_tpr /= len(classes)

        # Compute AUC
        mean_auc = auc(all_fpr, mean_tpr)

        # Calculate mean accuracy, precision, and specificity
        mean_acc = np.mean(acc_scores)
        mean_prec = np.mean(prec_scores)
        mean_spec = np.mean(spec_scores)

        # Prepare plotting
        ax.plot(all_fpr, mean_tpr, label='Mean ROC (area = {:.2f})'.format(mean_auc), color='b')
        ax.plot([0, 1], [0, 1], '--', label='Chance', color='grey')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve using threshold = {}'.format(input_threshold))

        # Add averaged metrics to the legend
        metrics_label = f'Accuracy: {mean_acc:.2f}, Precision : {mean_prec:.2f}, Specificity: {mean_spec:.2f}'
        ax.plot([], [], ' ', label=metrics_label)  # Using an empty plot as a workaround to add text to legend

        ax.legend(loc="lower right")

        # Draw the canvas and show the dialog
        canvas.draw()
        dialog.exec_()  # Use exec_() to make the dialog modal


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()