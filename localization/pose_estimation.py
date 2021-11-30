import math
import numpy as np
from utils import ARUCO_DICT, aruco_display
import cv2
import sys
import os
import time

from real_values import real_values


# Initialization
type = 'DICT_5X5_100'
aruco = None
calibration_matrix = np.load('../calibration/webcamera_creative/calibration_matrix.npy')
distortion_coefficients = np.load('../calibration/webcamera_creative/distortion_coefficients.npy')
video = cv2.VideoCapture(2)
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

record = True
if record:
    out = cv2.VideoWriter(os.path.join("../experiments/piwnica", 'video.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
    f = open(os.path.join("../experiments/piwnica", 'localization.csv'), "w")
    f.write(f"time,x,y,z,roll,pitch,yaw\n")

# Verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(type, None) is None:
    print(f"ArUCo tag type '{type}' is not supported")
    sys.exit(0)

while True:
    ret, image = video.read()

    if ret is False:
        break

    corners, ids, rejected = cv2.aruco.detectMarkers(image,
                                                     arucoDict,
                                                     parameters=arucoParams,
                                                     cameraMatrix=calibration_matrix,
                                                     distCoeff=distortion_coefficients)
    if len(corners) > 0:
        ids = ids.flatten()
        rv = np.empty((0, 3))
        pv = np.empty((0, 2))

        for (markerCorner, markerID) in zip(corners, ids):
            pv = np.vstack((pv, np.squeeze(markerCorner, axis=0)))
            try:
                rv = np.vstack((rv, real_values[markerID]))
            except KeyError:
                continue

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(rv,
                                                                        pv,
                                                                        calibration_matrix,
                                                                        distortion_coefficients,
                                                                        flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            continue

        Rt, _ = cv2.Rodrigues(rotation_vector)
        R = Rt.T
        pos = np.squeeze(-R @ translation_vector)
        roll = math.atan2(-R[2][1], R[2][2]) * 180 / math.pi
        pitch = math.asin(R[2][0]) * 180 / math.pi
        yaw = math.atan2(-R[1][0], R[0][0]) * 180 / math.pi

        detected_markers = aruco_display(corners, ids, rejected, image)
        msg_position = "[m]    x   : {0:.2f}".format(pos[0]) + "    y    : {0:.2f}".format(pos[1]) + "  z  : {0:.2f}".format(pos[2])
        msg_orientation = "[deg]  roll: {0:.2f}".format(roll) + "  pitch: {0:.2f}".format(pitch) + "  yaw: {0:.2f}".format(yaw)
        cv2.rectangle(image, (10, image.shape[0] - 80), (image.shape[1] - 10, image.shape[0] - 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, msg_position, (20, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, msg_orientation, (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Image preview", detected_markers)
        if record:
            out.write(detected_markers)
            f.write(f"{time.time()},{pos[0]},{pos[1]},{pos[2]},{roll},{pitch},{yaw}\n")

    else:
        cv2.rectangle(image, (10, image.shape[0] - 80), (165, image.shape[0] - 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, "not tracked", (20, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Image preview", image)
        if record:
            out.write(image)
            f.write(f"{time.time()},,,,,,\n")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if record:
    f.close()
    out.release()
cv2.destroyAllWindows()
video.release()
