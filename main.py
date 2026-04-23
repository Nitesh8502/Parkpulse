import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


ROOT = Path(__file__).resolve().parent

mask_path = str(ROOT / 'mask_1920_1080.png')
video_path = str(ROOT / 'data' / 'parking_1920_1080_loop.mp4')

if not os.path.exists(video_path):
    video_path = str(ROOT / 'samples' / 'parking_1920_1080_loop.mp4')


mask = cv2.imread(mask_path, 0)
if mask is None:
    raise FileNotFoundError(f'Unable to open mask file: {mask_path}')

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f'Unable to open video file: {video_path}')

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [False for _ in spots]
diffs = [0.0 for _ in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            previous_crop = previous_frame[y1:y1 + h, x1:x1 + w, :]
            if spot_crop.size > 0 and previous_crop.size > 0:
                diffs[spot_indx] = calc_diff(spot_crop, previous_crop)

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            safe_diffs = [float(d) if d is not None else 0.0 for d in diffs]
            max_diff = max(safe_diffs) if safe_diffs else 0
            if max_diff == 0:
                arr_ = []
            else:
                arr_ = [j for j in np.argsort(safe_diffs) if safe_diffs[j] / max_diff > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
