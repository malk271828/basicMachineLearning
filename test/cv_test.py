import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import pytest
from random import seed, random, randrange
seed(123)

from cv_util import *

@pytest.mark.parametrize("cmStr", ["jet", "spring", "copper"])
@pytest.mark.parametrize("mode", ["add", "overwrite"])
def test_image(cmStr, mode):
    n_sample = 255
    WIDTH, HEIGHT = 500, 300
    alpha = 0.01

    list_xy = list()
    for _ in range(n_sample):
        x = randrange(HEIGHT)
        y = randrange(WIDTH)
        cx = 100
        cy = 100
        list_xy.append((x, y, cx, cy, alpha))

    o, n, c, scaler = generateNormalizedPatchedImage(list_xy, shape=(WIDTH, HEIGHT), mode=mode, cmStr=cmStr, verbose=2)
    if scaler != None:
        print(scaler.data_max_)

@pytest.mark.parametrize("cmStr", ["jet", "copper"])
def test_group(cmStr):
    """
    Reference
    ---------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
    """
    n_group = 10
    n_sample = 255
    alpha = 0.01
    image = data.chelsea()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    WIDTH, HEIGHT = image.shape[:2]
    VIS_DIR = "visualization/"

    list_grouped_xy = list()
    for _ in range(n_group):
        list_xy = list()
        for _ in range(n_sample):
            x, y = randrange(HEIGHT), randrange(WIDTH)
            cx, cy = randrange(HEIGHT), randrange(WIDTH)
            list_xy.append((x, y, cx, cy, alpha))
        list_grouped_xy.append(list_xy)

    _, _, list_grouped_colored_array, _ = generateNormalizedGroupedPatchedImage(list_grouped_xy,
                                                                                shape=(WIDTH, HEIGHT),
                                                                                cmStr=cmStr,
                                                                                verbose=2)

    for i, colored_array in enumerate(list_grouped_colored_array):
        colored_array = np.reshape(colored_array, newshape=(-1, HEIGHT, 4))
        overlayed_array = (colored_array*128 + image/2.0).astype(np.uint8)
        overlayed_img = Image.fromarray(overlayed_array)
        overlayed_img.save(VIS_DIR + "group{0}_".format(i)+cmStr+"_overlayed.bmp")

    Image.fromarray((np.reshape(colored_array*128, (HEIGHT, WIDTH, 4))).astype(np.uint8)).save("colored_array.png")