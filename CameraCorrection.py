import numpy as np
import time
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import warnings
import pyrealsense2 as rs
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import os
from itertools import combinations, product

# 準備標定板
rows, cols = 8, 11
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# 儲存標定板的三維點和二維點
objpoints = []  # 三維點
imgpoints = []  # 二維點
image_list = []  # list of all images

# 擷取多張影像
images = glob.glob('.\\CheckBoardImageCorrection\\*')  # 放入你的影像路徑列表

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,11), None)

    if ret:
        image_list.append(img)
        objpoints.append(np.zeros((np.prod((8,11)), 3), dtype=np.float32))
        objpoints[-1][:, :2] = np.mgrid[0:(8,11)[0], 0:(8,11)[1]].T.reshape(-1, 2)*20

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (8,11), corners2, ret)
        cv2.imshow('iomg', img)
        filename_without_extension = os.path.splitext(fname)[0]
        full_path = os.path.normpath(os.path.join(os.getcwd(), filename_without_extension))
        now = os.getcwd()
        print(full_path+'_chessboard.png')
        cv2.imwrite(full_path+'_chessboard.png', img)
        cv2.waitKey(500)
        imgpoints.append(corners2)
    else:
        print(f"Chessboard corners not found in {fname}. Skipping this image.")
        imgpoints.append(None)

# Assuming you have object_points and image_points ready
# Perform camera calibration

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints,
                                                                    imagePoints=imgpoints,
                                                                    imageSize=(img.shape[1], img.shape[0]),
                                                                    cameraMatrix=None,
                                                                    distCoeffs=None)
transformation_matrices = []
common_transformation_matrix = np.eye(4)
for rvec, tvec in zip(rvecs, tvecs):
    # Compute the rotation matrix R and translation vector t from extrinsic parameters
    R, _ = cv2.Rodrigues(rvec)

    # Construct the transformation matrix T (4x4 homogeneous transformation matrix)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.T

    # Append the transformation matrix to the list
    transformation_matrices.append(T)

    # Compute the combined transformation matrix
    for T in transformation_matrices:
        common_transformation_matrix = np.dot(common_transformation_matrix, T)

        # output would be a transformation matrix to common world coordinate system
    # output would be a transformation matrix to common world coordinate system
# 儲存校正參數到 YAML 文件
calibration_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist(),
}

with open('calibration.yaml', 'w') as file:
    yaml.dump(calibration_data, file)

