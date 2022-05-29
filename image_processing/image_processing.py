"""
Main script for processing an image:
it preprocesses the image, detects the line, estimate line curve
and do the path planning
"""
from __future__ import print_function, with_statement, division, absolute_import

import argparse

import cv2
import numpy as np

from constants import REF_ANGLE, MAX_ANGLE, EXIT_KEYS, WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE, TARGET_POINT
from path_planning.bezier_curve import computeControlPoints, bezier

# Load trained model
# model = loadNetwork(WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE)


def processImage(image, debug=False):
    """
    :param image: (bgr image)
    :param debug: (bool)
    :return:(float, float)
    """
    x, y = processRois(image)
    # x, y = predict(model, image)
    if debug:
        return x, y

    # Compute bezier path and target point
    control_points = computeControlPoints(x, y, add_current_pos=True)
    target = bezier(TARGET_POINT, control_points)

    # Linear Regression to fit a line
    # It estimates the line curve

    # Case x = cst, m = 0
    if len(np.unique(x)) == 1: # pragma: no cover
        turn_percent = 0
    else:
        # Linear regression using least squares method
        # x = m*y + b -> y = 1/m * x - b/m if m != 0
        A = np.vstack([y, np.ones(len(y))]).T
        m, b = np.linalg.lstsq(A, x, rcond=-1)[0]

        # Compute the angle between the reference and the fitted line
        track_angle = np.arctan(1 / m)
        diff_angle = abs(REF_ANGLE) - abs(track_angle)
        # Estimation of the line curvature
        turn_percent = (diff_angle / MAX_ANGLE) * 100
    return turn_percent, target[0]


def processImageOpenCV(image, debug=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 99, 0])
    upper_white = np.array([179, 255, 87])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel_erode = np.ones((4, 4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6, 6), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    x = 0
    y = 0
    if len(contours) > 0:
        M = cv2.moments(contours[0])
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
    return x, y

def processRois(image, roiHeight = 40):
    shape = image.shape
    height = shape[0]
    width = shape[1]

    roi1 = image[0:roiHeight, 0:width]
    roi2 = image[height // 2 - roiHeight // 2:(height // 2 - roiHeight // 2) + 40, 0:width]
    roi3 = image[height - roiHeight:height, 0:width]

    x1, y1 = processImageOpenCV(roi1)
    x2, y2 = processImageOpenCV(roi2)
    x3, y3 = processImageOpenCV(roi3)

    cv2.imshow("roi1", roi1)
    cv2.imshow("roi2", roi2)
    cv2.imshow("roi3", roi3)

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    return x, y
