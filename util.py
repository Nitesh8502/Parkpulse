import pickle
from pathlib import Path
from sklearn import __version__ as sklearn_version

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.p"
EXPECTED_SKLEARN_VERSION = "1.1.3"

if sklearn_version != EXPECTED_SKLEARN_VERSION:
    raise RuntimeError(
        "Incompatible scikit-learn version for model.p. "
        f"Expected {EXPECTED_SKLEARN_VERSION}, found {sklearn_version}. "
        "Use the Python 3.10 virtual environment at .venv310 for accurate predictions."
    )

with open(MODEL_PATH, "rb") as model_file:
    MODEL = pickle.load(model_file)


def empty_or_not(spot_bgr):

    if spot_bgr is None or spot_bgr.size == 0:
        return NOT_EMPTY

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    if not np.isfinite(img_resized).all():
        return NOT_EMPTY
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)
    prediction = int(y_output[0])

    if prediction == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

