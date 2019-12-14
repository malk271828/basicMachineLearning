import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from random import seed, random, randrange
seed(123)

from cv_util import generateNormalizedPatchedImage

def test_image():
    n_sample = 20
    WIDTH, HEIGHT = 500, 300
    alpha = 10

    list_xy = list()
    for _ in range(n_sample):
        x = randrange(WIDTH)
        y = randrange(HEIGHT)
        cx = 100
        cy = 100
        list_xy.append((x, y, cx, cy, alpha))

    generateNormalizedPatchedImage(list_xy, WIDTH, HEIGHT, verbose=1)
