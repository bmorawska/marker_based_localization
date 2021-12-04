import math
import numpy as np
from utils import ARUCO_DICT, aruco_display
import cv2
import sys
import os
import pickle
import yaml

from real_values import real_values

# Loading settings file
with open("../settings/settings.yaml", "r") as settings_stream:
    try:
        settings = yaml.safe_load(settings_stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(0)

# Initialization
type = settings['aruco']['type']
calibration_matrix = np.load(os.path.join('..', settings['image']['calibration_matrix']))
distortion_coefficients = np.load(os.path.join('..', settings['image']['distortion_matrix']))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

rpy_image = cv2.imread('../rpy.png', cv2.IMREAD_UNCHANGED)
rpy_image_x1, rpy_image_y1, rpy_image_x2, rpy_image_y2 = 10, 10, rpy_image.shape[1] + 10, rpy_image.shape[0] + 10

image_path = os.path.join(os.path.join('..', settings['input']['photo']['input_dir']), settings['input']['photo']['filename'])

load_pickle = True
if load_pickle:
    with open(os.path.join('..', settings['map']['pickle_path']), 'rb') as f:
        real_values = pickle.load(f)
        print("Loaded configuration from file.")

if ARUCO_DICT.get(type, None) is None:
    print(f"ArUCo tag type '{type}' is not supported")
    sys.exit(0)

# Image processing
image = cv2.imread(image_path)
if image is None:
    print(f"Cannot read photo {image_path}")
    sys.exit(0)

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
            cv2.destroyAllWindows()
            print("KeyError problem.")
            sys.exit(0)

    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(rv,
                                                                    pv,
                                                                    calibration_matrix,
                                                                    distortion_coefficients,
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        cv2.destroyAllWindows()
        print(f"Cannot read photo {image_path}")
        sys.exit(0)

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
    image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] = \
        image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] * (1 - rpy_image[:, :, 3:] / 255) + \
        rpy_image[:, :, :3] * (rpy_image[:, :, 3:] / 255)
    cv2.imshow("Image preview", detected_markers)

else:
    cv2.rectangle(image, (10, image.shape[0] - 80), (165, image.shape[0] - 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, "not tracked", (20, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Image preview", image)

# Finishing kindly
cv2.waitKey(0)
cv2.destroyAllWindows()
