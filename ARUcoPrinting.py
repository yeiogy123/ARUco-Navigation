import cv2
import cv2.aruco as aruco
# 設置 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 設置標記 ID
marker_id = 0  # 這裡替換為你想生成的 ArUco 標記的 ID
# 設置標記大小和邊框大小
marker_size = 300
# 生成 ArUco 標記圖片
marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
#cv2.imwrite(".\\ARUcoIMG\\marker_{}.png".format(marker_id), marker_image)

#marker_img = cv2.imread(".\\ARUcoIMG\\marker_{}.png".format(marker_id))

#cv2.imshow("ArUco Marker", marker_img)

#print("Dimensions:", marker_img.shape)

#cv2.waitKey(0)

#cv2.destroyAllWindows()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters(adaptiveThreshWinSizeMin = 2)