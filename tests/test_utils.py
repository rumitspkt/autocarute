from __future__ import print_function, division, absolute_import

import numpy as np

from .common import *
from train.utils import preprocessImage, loadNetwork, predict, loadLabels
from constants import *

test_image = 234 * np.ones((MAX_WIDTH, MAX_HEIGHT, 3), dtype=np.uint8)


def testPreprocessing():
    image = preprocessImage(test_image, INPUT_WIDTH, INPUT_HEIGHT)
    # Check normalization
    assertEq(len(np.where(np.abs(image) > 1)[0]), 0)
    # Check resize
    assertEq(image.size, 3 * INPUT_WIDTH * INPUT_HEIGHT)
    # Check normalization on one element
    assertEq(image[0, 0, 0], ((234 / 255.) - 0.5) * 2)


def testPredict():
    model = loadNetwork(WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE)
    x, y = predict(model, test_image)


def testLoadLabels():
    loadLabels(DATASET)
    dataset_no_trailing_slash = DATASET[:-1]
    folders = [DATASET, dataset_no_trailing_slash]
    loadLabels(folders)
