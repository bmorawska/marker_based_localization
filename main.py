import base64
import math
import numpy as np
import cv2
import sys
import os
import time
import statistics
import uuid
from playsound import playsound
import json
from datetime import datetime
import socketio

from utils import ARUCO_DICT, aruco_display


def RotMatrix(alpha, beta, gamma):
    a2pi = np.pi / 180
    alpha *= a2pi
    beta *= a2pi
    gamma *= a2pi

    M = np.array([
        [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
         np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],
        [np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
         np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)],
        [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)]
    ])
    return M


def load_settings():
    with open(os.path.join('settings', 'settings.json'), "r") as read_file:
        settings = json.load(read_file)
    return settings


def load_markers():
    with open(os.path.join('settings', 'markers.json'), "r") as read_file:
        markers = json.load(read_file)
    real_values = {}
    for aruco in markers['aruco']:
        real_values[aruco['id']] = {"pos": [aruco["position"]["x"], aruco["position"]["y"], aruco["position"]["z"]],
                                    "rot": aruco["rotation"]}
    return real_values


# Map initialization
real_values = load_markers()

# Settings initialization
settings = load_settings()

# Camera initialization
type = settings['aruco']['type']
if ARUCO_DICT.get(type, None) is None:
    print(f"ArUCo tag type '{type}' is not supported")
    sys.exit(0)
calibration_matrix = np.load(settings['camera']['calibration_matrix'])
distortion_coefficients = np.load(settings['camera']['distortion_matrix'])
video = cv2.VideoCapture(settings['camera']['id'])
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Display initialization
prev_frame_time = 0
new_frame_time = 0

rpy_image = cv2.imread(os.path.join('assets', 'rpy.png'), cv2.IMREAD_UNCHANGED)
rpy_image_x1, rpy_image_y1, rpy_image_x2, rpy_image_y2 = 10, 10, rpy_image.shape[1] + 10, rpy_image.shape[0] + 10

# Recording initialization
record = settings['movie']['status']
if record:
    if not os.path.exists('movies'):
        os.makedirs('movies')

    out = cv2.VideoWriter(os.path.join('movies', settings['movie']['filename']),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          settings['movie']['fps'],
                          (settings['movie']['resolution'][0], settings['movie']['resolution'][1]))

    f = open(os.path.join('movies', settings['movie']['filename'].split('.')[0] + '.csv'), "w")
    f.write(f"time,x,y,z,roll,pitch,yaw\n")

# Snapshot initialization
snapshot = settings['snapshot']['status']
if snapshot:
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')

    if not os.path.exists(os.path.join('snapshots', 'snapshot.csv')):
        snapshot_file = open(os.path.join('snapshots', 'snapshot.csv'), "a")
        snapshot_file.write(f"key,time,x,y,z,roll,pitch,yaw\n")
    else:
        snapshot_file = open(os.path.join('snapshots', 'snapshot.csv'), "a")

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')


@sio.event
def disconnect():
    print('disconnected from server')


def emit_anchors():
    anchors = []
    for key in real_values:
        anchors.append({'name': f'#{key}', 'x': float(real_values[key]['pos'][0]), 'y': float(real_values[key]['pos'][1])})

    sio.emit('anchors', {'anchors': anchors})

websocket = settings['websocket']['status']
if websocket:
    sio.connect(f'http://127.0.0.1:3000')
    emit_anchors()
    time_elapsed = time.time()
    send_frequency = 1 / settings['websocket']['frequency']


if __name__ == '__main__':
    # Camera loop
    while True:
        ret, image = video.read()
        if ret is False:
            break
        new_frame_time = time.time()
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
            areas = []
            indices = []
            # For every aruco
            for i in range(0, len(ids)):
                if not ids[i] in list(real_values.keys()):
                    continue
                rotation_vector, translation_vector, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],
                                                                                                        settings[
                                                                                                            'aruco'][
                                                                                                            'size'],
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

                P = np.array([pos[0], pos[1], pos[2]])
                M = RotMatrix(0, -real_values[ids[i][0]]['rot'], 0)
                P = M @ P

                px = P[0] + real_values[ids[i][0]]['pos'][0]
                py = P[2] + real_values[ids[i][0]]['pos'][1]
                pz = P[1] + real_values[ids[i][0]]['pos'][2]

                rolls.append(roll)
                pitches.append(pitch)
                yaws.append(yaw)

                xs.append(px)
                ys.append(py)
                zs.append(pz)
                ax0, ax1, ax2, ax3 = corners[i][0][0][0], corners[i][0][1][0], corners[i][0][1][0], corners[i][0][1][0]
                ay0, ay1, ay2, ay3 = corners[i][0][0][1], corners[i][0][1][1], corners[i][0][1][1], corners[i][0][1][1]
                area = abs((ax0 * ay1 - ay0 * ax1) + (ax1 * ay2 - ay1 * ax2) + (ax2 * ay3 - ay2 * ax3) + (
                        ax3 * ay0 - ay0 * ax3))
                areas.append(area)
                indices.append(ids[i])

            if len(xs) == 0:
                continue
            areas = np.array(areas)
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)
            total_area = sum(areas)
            areas /= total_area
            xs *= areas
            ys *= areas
            zs *= areas
            x = sum(xs)
            y = sum(ys)
            z = sum(zs)

            roll = statistics.mean(rolls)
            pitch = statistics.mean(pitches)
            yaw = statistics.mean(yaws)
            if websocket:
                sio.emit('position', {'x': x, 'y': y})

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Display results
            detected_markers = aruco_display(corners, ids, rejected, image)
            msg_position = "[m]    x   : {0:.2f}".format(x) + "    y    : {0:.2f}".format(y) \
                           + "  z  : {0:.2f}".format(z)
            msg_orientation = "[deg]  roll: {0:.2f}".format(roll) + "  pitch: {0:.2f}".format(
                pitch) + "  yaw: {0:.2f}".format(
                yaw)
            cv2.rectangle(image, (10, image.shape[0] - 80), (image.shape[1] - 10, image.shape[0] - 10), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(image, msg_position, (20, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
            cv2.putText(image, msg_orientation, (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(image, f"{int(fps)} fps", (image.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] = \
                image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] * (1 - rpy_image[:, :, 3:] / 255) + \
                rpy_image[:, :, :3] * (rpy_image[:, :, 3:] / 255)
            cv2.imshow("ArUco", detected_markers)
            if record:
                out.write(detected_markers)
                f.write(f"{datetime.now()},{x},{y},{z},{roll},{pitch},{yaw}\n")
            if snapshot:
                snapshot_msg = f"{datetime.now()},{x},{y},{z},{roll},{pitch},{yaw}\n"
            if websocket:
                dur = time.time() - time_elapsed
                if dur > send_frequency:
                    time_elapsed = time.time()
                    res, sio_frame = cv2.imencode('.jpg', detected_markers)  # from image to binary buffer
                    sio_data = base64.b64encode(sio_frame)  # convert to base64 format
                    sio.emit('frame', sio_data)

        else:
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.rectangle(image, (10, image.shape[0] - 80), (165, image.shape[0] - 10), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, "not tracked", (20, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(image, f"{int(fps)} fps", (image.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] = \
                image[rpy_image_y1:rpy_image_y2, rpy_image_x1:rpy_image_x2] * (1 - rpy_image[:, :, 3:] / 255) + \
                rpy_image[:, :, :3] * (rpy_image[:, :, 3:] / 255)
            cv2.imshow("ArUco", image)
            if record:
                out.write(image)
                f.write(f"{datetime.now()},,,,,,\n")
            if snapshot:
                snapshot_msg = f"{datetime.now()},,,,,,\n"
            if websocket:
                dur = time.time() - time_elapsed
                if dur > send_frequency:
                    time_elapsed = time.time()
                    res, sio_frame = cv2.imencode('.jpg', image)  # from image to binary buffer
                    sio_data = base64.b64encode(sio_frame)  # convert to base64 format
                    sio.emit('frame', sio_data)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            playsound(os.path.join('assets', 'shutter_sound.wav'), False)
            keyname = uuid.uuid4()
            snapshot_file.write(f"{keyname},")
            snapshot_file.write(snapshot_msg)
            cv2.imwrite(os.path.join('snapshots', f"{keyname}.png"), image)
            print(f'[{keyname}] Snapshot saved: {snapshot_msg}')

    # Finishing kindly
    if record:
        f.close()
        out.release()
    if snapshot:
        snapshot_file.close()
    cv2.destroyAllWindows()
    video.release()
    sio.disconnect()
