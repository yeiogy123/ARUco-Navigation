import math
import os
import re
import sys

import numpy
import yaml

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, \
    QFileDialog, QTableWidget, QTableWidgetItem, QScrollArea
from PyQt5.QtGui import QPixmap, QFont
import cv2

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ARUcoNavigation")
        self.setGeometry(100, 100, 1300, 900)

        # Buttons
        self.select_image_button = QPushButton("選擇圖片")
        self.select_image_button.clicked.connect(self.select_image)

        self.execute_function_button = QPushButton("測量相機中心到ARUco中心的距離")
        self.execute_function_button.clicked.connect(self.execute_function)

        # Image display labels
        self.image_label1 = QLabel()
        self.parameter = QLabel()

        # Text display
        self.text_left_label = QLabel()
        self.text_right_label = QLabel()
        self.text_edit_left = QTextEdit()
        self.text_edit_left.setReadOnly(True)
        self.text_edit_right = QTextEdit()
        self.text_edit_right.setReadOnly(True)

        self.t = QLabel()
        self.D = QLabel()
        self.R = QLabel()
        self.T = QLabel()
        self.text_label = QLabel()
        self.text_left_unit = QLabel()
        self.text_right_uint = QLabel()

        self.image_path = ""
        self.actual_distance = 0
        # Layout
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_image_button)
        button_layout.addWidget(self.execute_function_button)


        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label1)
        image_layout.addWidget(self.parameter)
        text_layout = QHBoxLayout()
        text_layout.addWidget(self.text_left_label)
        text_layout.addWidget(self.text_edit_left)
        text_layout.addWidget(self.text_left_unit)
        text_layout.addWidget(self.text_right_label)
        text_layout.addWidget(self.text_edit_right)
        text_layout.addWidget(self.text_right_uint)
        text_label_layout = QHBoxLayout()
        text_label_layout.addWidget(self.D)
        text_label_layout.addWidget(self.t)
        vector_layout = QHBoxLayout()
        vector_layout.addWidget(self.R)
        vector_layout.addWidget(self.T)
        #text_label_layout.addWidget(self.text_label)
        #text_layout.addWidget(self.text_label)
        self.text_edit_left.setMaximumSize(200, 50)
        self.text_edit_right.setMaximumSize(200, 50)
        self.parameter.setMaximumSize(640,480)
        font = QFont()
        font.setPointSize(20)  # 设置字体大小为 12
        self.parameter.setFont(font)
        self.text_left_label.setFont(font)
        self.text_left_unit.setFont(font)
        self.text_right_label.setFont(font)
        self.text_right_uint.setFont(font)
        self.text_left_label.setText("相機中心到ARUco中心的實際距離 ： ")
        self.text_left_unit.setText('mm')
        self.text_right_label.setText("相機中心到ARUco中心的量測距離 ： ")
        self.text_right_uint.setText('mm')
        self.D.setFont(font)
        self.t.setFont(font)
        self.R.setFont(font)
        self.T.setFont(font)

        self.text_label.setFont(font)
        self.text_label.setText("Camera : Intel RealSense D435if\n"
                                "Photo : RGB\n"
                                "1 Pixels = 0.0014 mm\n"
                                "Resolution : 640 × 480\n"
                                "FOV : 87° × 58°")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(5)
        self.table_widget.setColumnCount(2)
        self.table_widget.setColumnWidth(1, 150)

        #self.table_widget.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.table_widget.horizontalHeader().setVisible(False)
        self.table_widget.verticalHeader().setVisible(False)
        self.spec_table()
        self.scroll_area.setWidget(self.table_widget)

        # Limit the size of the scroll area
        self.scroll_area.setMaximumSize(255,160)

        # Layout
        layout = QVBoxLayout()


        #self.text_label.setMaximumSize(640, 100)
        layout.addLayout(button_layout)
        layout.addLayout(image_layout)
        layout.addLayout(text_layout)
        layout.addLayout(text_label_layout)
        layout.addLayout(vector_layout)
        layout.addWidget(self.scroll_area)
        camera_matrix_text = "608.6, 0.0, 329.2,\n" \
                             "0.0,  608.5, 239.2,\n" \
                             "0.0,   0.0, 1.0"

        dist_coeff_text = "[- 0.0138, 0.6504, 0.0008,- 0.0006, - 2.1840]"

        self.parameter.setText(
            f"Intrinsic Parameter : \n{camera_matrix_text}\n\nDistortion Coefficients : \n{dist_coeff_text}")

        self.setLayout(layout)
        self.setLayout(layout)
    def spec_table(self):
        attributes = ["Camera", "Photo", "1 Pixels", "Resolution", "FOV"]
        values = ["Intel RealSense D435if", "RGB",  "0.0014 mm", "640 × 480", "87° × 58°"]

        for i, (attribute, value) in enumerate(zip(attributes, values)):
            self.table_widget.setItem(i, 0, QTableWidgetItem(attribute))
            self.table_widget.setItem(i, 1, QTableWidgetItem(value))
    def transformFromCTA(self, r, t):
        R, _ = cv2.Rodrigues(r)
        t = np.squeeze(t)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        C = np.array([0,0,0,1], dtype=np.float32)
        Ti = np.linalg.inv(T)
        point = [Ti @ C][:3]
        c = np.array([0,0,0,1], dtype=np.float32)
        test = [T @ c]
        print(test)
        print('aruco center in camera coordinate:', test)
        return test
    def select_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "選擇圖片", "", "Images (*.png *.jpg)")
        if filename:
            pixmap = QPixmap(filename)
            self.image_label1.setPixmap(pixmap)
            self.image_path = filename
    def detect_target_aruco(self, image):
        if not os.path.exists('.\\calibration.yaml'):
            print('File does not exist')
        else:
            try:
                with open('calibration.yaml') as f:
                    calib = yaml.load(f, Loader=yaml.FullLoader)
                    camera_coeff = calib['camera_matrix']
                    dist_coeff = calib['dist_coeff']
                    camera_matrix = np.array(camera_coeff, dtype=np.float32)
                    dist_matrix = np.array(dist_coeff, dtype=np.float32)
                    color_image = cv2.imread(image)

                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (color_image.shape[1], color_image.shape[0]), 0, (color_image.shape[1], color_image.shape[0]))
                    dist = cv2.undistort(color_image, camera_matrix, dist_matrix, None, new_camera_matrix)
                    x, y, w, h = roi
                    dist = dist[y:y + h, x:x + w]
                    color_image = dist
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    parameters = cv2.aruco.DetectorParameters()
                    corners, marker_id, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
                    height, width, _ = color_image.shape
                    middle_x = math.floor(int(color_image.shape[0]) / 2)
                    middle_y = math.floor(int(color_image.shape[1]) / 2)
                    cv2.line(color_image, (middle_y - 10,middle_x), (middle_y + 10,middle_x), (0, 0, 255), 2)
                    cv2.line(color_image, (middle_y, middle_x - 10), (middle_y, middle_x + 10), (0, 0, 255), 2)
                    if marker_id is not None:
                        rotation_vector, translation_vector, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 50, camera_matrix, dist_matrix)
                        (rotation_vector-translation_vector).any()
                        for i in range(rotation_vector.shape[0]):
                            cv2.aruco.drawDetectedMarkers(color_image, corners, marker_id)
                            cv2.drawFrameAxes(color_image, camera_matrix, dist_matrix, rotation_vector[i, :, :],
                                              translation_vector[i, :, :], 15)
                        print(markerPoints)
                        test = self.transformFromCTA(rotation_vector, translation_vector)
                        print('------------------------------------')
                        print(test[0][0],'mm')
                        print('------------------------------------')
                        print(self.image_path[:-4] + "_Result.png")
                        result_image_path = self.image_path[:-4] + "_Result.png"  # 將 '.png' 替換為 '_Result.png'
                        cv2.imwrite(result_image_path,color_image)
                        return result_image_path, test[0][0], rotation_vector, translation_vector
                        #ToDo:Translation_vector change into the depth camera distance
                    else:
                        cv2.putText(color_image, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except cv2.error as e:
                print("An error occurred while opening the file:", e)

    def execute_function(self):
        # 在這裡執行您的函數，並處理影像和文本輸出
        # 這裡只是一個示例，您需要根據您的實際情況進行修改
        print("image:",self.image_path)
        output, distance, r, t= self.detect_target_aruco(self.image_path)
        print(output)
        text_output = distance
        print('distance:', distance)
        pixmap = QPixmap(output)
        actual_length = re.search(r'(\d+)', output).group(1)
        print(actual_length)
        self.actual_distance = actual_length
        self.image_label1.setPixmap(pixmap)
        self.text_edit_left.setFontPointSize(20)
        self.text_edit_right.setFontPointSize(20)
        self.text_edit_left.setPlainText(f"{self.actual_distance}")
        self.text_edit_left.setAlignment(Qt.AlignRight)  # 导入 `Qt` 模块
        self.text_edit_right.setPlainText(f"{'{:.4f}'.format(abs(text_output))}")
        self.text_edit_right.setAlignment(Qt.AlignRight)  # 导入 `Qt` 模块
        print(float(actual_length),float(distance))
        distance_error = abs(float(distance)) - float(actual_length)
        print(distance_error)
        print('R=', numpy.asarray(r), 'T : ', numpy.asarray(t))
        rotation_text = "{}".format(np.squeeze(r))
        translation_text = "{}".format(np.squeeze(t))
        print(rotation_text, translation_text)
        print(self.image_path)
        print(os.path.dirname(self.image_path))
        print(os.path.basename(os.path.dirname(self.image_path)))
        Depth = os.path.basename(os.path.dirname(self.image_path))
        if Depth == 'hundred':
            d = 1000
        elif Depth == 'hundred_fourty':
            d = 1400
        elif Depth == 'hundred_sixty':
            d = 1600
        else:
            d = 30
        self.D.setText(f"Depth Distance : {d}mm")
        self.t.setText(f"x軸誤差 : {'{:.2f}'.format(abs(distance_error))} mm")
        self.R.setText(f"Rotation Vector : {rotation_text}")
        self.T.setText(f"Translation Vector : {translation_text}")
        print('1 pixel = 0.0014mm')
        print('success')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
