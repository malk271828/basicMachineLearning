import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
from random import seed, random, randrange
seed(123)

from cv_util import *

@pytest.mark.parametrize("cmStr", ["jet", "spring", "plasma"])
def test_image(cmStr):
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

    o, n, c, scaler = generateNormalizedPatchedImage(list_xy, shape=(WIDTH, HEIGHT), cmStr=cmStr, verbose=2)
    print(scaler.data_max_)

@pytest.mark.parametrize("cmStr", ["jet"])
def test_group(cmStr):
    n_group = 2
    n_sample = 255
    WIDTH, HEIGHT = 500, 300
    alpha = 0.01

    list_grouped_xy = list()
    for _ in range(n_group):
        list_xy = list()
        for _ in range(n_sample):
            x = randrange(HEIGHT)
            y = randrange(WIDTH)
            cx = 100
            cy = 100
            list_xy.append((x, y, cx, cy, alpha))
        list_grouped_xy.append(list_xy)

    generateNormalizedGroupedPatchedImage(list_grouped_xy, shape=(WIDTH, HEIGHT), cmStr=cmStr, verbose=2)
