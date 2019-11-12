import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest

from featureExtractor import *

@pytest.fixture
def expFixture():
    class _expFixture:
        def __init__(self):
            self.fileID = "eIu0CXKekI4"
            self.filePath = self.fileID + ".mp4"

            # The following message indicates version mismatch between code and model.
            # RuntimeError: Unexpected version found while deserializing dlib::shape_predictor.
            # https://stackoverflow.com/questions/49614460/python-unexpected-version-found-while-deserializing-dlibshape-predictor
            self.SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    return _expFixture()

def test_dataFetch(expFixture):
    """fetch video data and extract raw audio (pcm)
        play raw audio file the following command:
        >>> aplay -f S16_LE -c2 -r22050 soundfile.raw
    """
    from pytube import YouTube

    yt = YouTube("https://youtu.be/"+expFixture.fileID)
    if not os.path.exists(expFixture.fileID+".mp4"):
        print("video file is not found")
        yt.streams.first().download(".")
        os.rename(yt.title+".mp4", expFixture.fileID+".mp4")

    subprocess.call(["ffmpeg -i {0}.mp4 -vn -f s16le -acodec pcm_s16le soundfile.raw".format(expFixture.fileID)], shell=True, cwd=".")

def test_landmarks(expFixture):
    # w/o specifying cache path
    le1 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath)

    # specifying cache path
    le2 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath, cache_dir="../cache2/")

    landmarks_list = le1.getLandmarks(verbose=1)
    landmarks_list = le2.getLandmarks(verbose=1)

def test_batch(expFixture):
    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, [expFixture.filePath]*3)
    X = be.getX(verbose=2)
    print(X.shape)

    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, [expFixture.filePath]*3)
    X = be.getX(verbose=2)
    print(X.shape)
