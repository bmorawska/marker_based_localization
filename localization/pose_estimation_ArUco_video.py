import math
import numpy as np
from utils import ARUCO_DICT, aruco_display
import cv2
import sys
import os
import time
import pickle
import yaml
import statistics

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
video = cv2.VideoCapture(settings['input']['camera']['id'])
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

rpy_image = cv2.imread('../rpy.png', cv2.IMREAD_UNCHANGED)
rpy_image_x1, rpy_image_y1, rpy_image_x2, rpy_image_y2 = 10, 10, rpy_image.shape[1] + 10, rpy_image.shape[0] + 10

load_pickle = settings['map']['load_pickle']
if load_pickle:
    with open(os.path.join('..', settings['map']['pickle_path']), 'rb') as f:
        real_values = pickle.load(f)
        print("Loaded configuration from file.")

record = settings['results']['record']
if record:
    out = cv2.VideoWriter(os.path.join(os.path.join('..', settings['results']['output_dir']),
                                       settings['results']['movie']['movie_filename']),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          settings['results']['movie']['fps'],
                          (settings['results']['movie']['resolution']['width'],
                           settings['results']['movie']['resolution']['height'],))

    f = open(os.path.join(os.path.join('..', settings['results']['output_dir']), settings['results']['csv_filename']), "w")
    f.write(f"time,x,y,z,roll,pitch,yaw\n")


if ARUCO_DICT.get(type, None) is None:
    print(f"ArUCo tag type '{type}' is not supported")
    sys.exit(0)

# Camera loop
while True:
    ret, image = video.read()

    if ret is False:
        break

    corners, ids, rejected = cv2.aruco.detectMarkers(image,
                                                     arucoDict,
                                                     parameters=arucoParams,
                                                     cameraMatrix=calibration_matrix,
                                                     distCoeff=distortion_coefficients)
    # If ArUco detected in frame
    if len(corners) > 0:
        rolls = []
        pitches = []
        yaws = []
        xs = []
        ys = []
        zs = []
        for i in range(0, len(ids)):
            rotation_vector, translation_vector, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],
                                                                                                    settings['aruco']['size'],
                                                                                                    calibration_matrix,
                                                                                                    distortion_coefficients)

            Rt, _ = cv2.Rodrigues(rotation_vector)
            R = Rt.T
            pos = -R @ np.squeeze(translation_vector)
            roll = math.atan2(-R[2][1], R[2][2]) * 180 / math.pi
            if roll < 0:
                roll = -roll - 180
            else:
                roll = -roll + 180
            roll = -roll
            pitch = -math.asin(R[2][0]) * 180 / math.pi
            yaw = -math.atan2(-R[1][0], R[0][0]) * 180 / math.pi + 90
            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

        x = statistics.mean(xs)
        y = statistics.mean(ys)
        z = statistics.mean(zs)
        roll = statistics.mean(rolls)
        pitch = statistics.mean(pitches)
        yaw = statistics.mean(yaws)

        # Display results
        detected_markers = aruco_display(corners, ids, rejected, image)
        msg_position = "[m]    x   : {0:.2f}".format(x) + "    y    : {0:.2f}".format(y) \
                       + "  z  : {0:.2f}".format(z)
        msg_orientation = "[deg]  roll: {0:.2f}".format(roll) + "  pitch: {0:.2f}".format(
            pitch) + "  yaw: {0:.2f}".format(
            yaw)
        cv2.rectangle(image, (10, image.shape[0] - 80), (image.shape[1] - 10, image.shape[0] - 10), (0, 0, 0),cv2.FILLED)
        cv2.putText(image, msg_position, (20, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, msg_orientation, (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
        image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] = \
            image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] * (1 - rpy_image[:, :, 3:] / 255) + \
            rpy_image[:, :, :3] * (rpy_image[:, :, 3:] / 255)
        cv2.imshow("Image preview", detected_markers)
        if record:
            out.write(detected_markers)
            f.write(f"{time.time()},{x},{y},{z},{roll},{pitch},{yaw}\n")

    else:
        cv2.rectangle(image, (10, image.shape[0] - 80), (165, image.shape[0] - 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, "not tracked", (20, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] = \
            image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] * (1 - rpy_image[:, :, 3:] / 255) + \
            rpy_image[:, :, :3] * (rpy_image[:, :, 3:] / 255)
        cv2.imshow("Image preview", image)
        if record:
            out.write(image)
            f.write(f"{time.time()},,,,,,\n")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Finishing kindly
if record:
    f.close()
    out.release()
cv2.destroyAllWindows()
video.release()