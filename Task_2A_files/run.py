import cv2
from cv2 import aruco
img_dir_path = "public_test_cases/"
img_file_path = img_dir_path +  'aruco_' + str(0) + '.png'
image = cv2.imread(img_file_path)
# bw_img = cv2.cvtColor(img,7)
# print(bw_img)
ArUco_details_dict = {}
ArUco_corners = {}

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

print(aruco_dict)
print(parameters)

# Detect ArUco markers in the image
corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
print(corners)
print(ids)

if ids is not None:
    for i in range(len(ids)):
        marker_id = ids[i][0]  # Get the marker's ID
        marker_corners = corners[i][0]  # Get the corner coordinates

        # Calculate the center of the marker
        center_x = int((marker_corners[:, 0].sum()) / 4)
        center_y = int((marker_corners[:, 1].sum()) / 4)

        # Calculate the angle from the vertical axis
        # You may need to customize this angle calculation based on your needs
        angle = 0  # Replace with your angle calculation logic

        # Update the dictionaries with marker details
        ArUco_details_dict[marker_id] = [[center_x, center_y], angle]
        ArUco_corners[marker_id] = marker_corners
print(ArUco_details_dict)
print(ArUco_corners)

