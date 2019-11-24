import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam
import mlflow
import mlflow.sklearn
import mlflow.keras

from landmarkExtractor import *
from lombardFileSelector import *
from vae import *

@pytest.fixture
def expFixture():
    class _expFixture:
        def __init__(self):
            fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")
            self.filePath = fileSelector.getFileList("visual")

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

@pytest.mark.landmark
def test_landmarks(expFixture):
    # w/o specifying cache path
    le1 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath)

    # specifying cache path
    le2 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath, cache_dir="../cache2/")

    landmarks_list = le1.getLandmarks(verbose=1)
    landmarks_list = le2.getLandmarks(verbose=1)

@pytest.mark.landmark
def test_batch(expFixture):
    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath)
    X = be.getX(verbose=2)
    print(X.shape)

    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath)
    X = be.getX(verbose=2)
    print(list(X[0]))

    assert X[0][DLIB_CENTER_INDEX*2] == 0 and X[0][DLIB_CENTER_INDEX*2+1] == 0

def test_lombardFileSelector():
    fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")

    fileSelector.getFileList("audio", verbose=1)
    fileSelector.getFileList("visual", verbose=1)

def test_mouth_open(expFixture):
    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath[1:2])
    X = be.getX(verbose=2)

    upperLipY = pd.DataFrame([landmarks[DLIB_UPPERLIP_INDEX*2+1] for landmarks in X])
    lowerLipY = pd.DataFrame([landmarks[DLIB_LOWERLIP_INDEX*2+1] for landmarks in X])

    df = pd.concat([upperLipY, lowerLipY, lowerLipY - upperLipY], axis=1)
    df.columns=["upperLipY", "lowerLipY", "MouthOpenLength"]
    df.plot()
    plt.show()

@pytest.mark.vae
def test_vae_train(expFixture):
    enable_mse = False
    enable_graph = False
    latent_dim = 64
    batch_size = 128
    epochs = 100
    be = batchExtractor(expFixture.SHAPE_PREDICTOR_PATH, expFixture.filePath)
    X = be.getX(verbose=2)
    input_shape = X[0].shape

    scaler = MinMaxScaler()
    vae, encoder, decoder, vae_loss = build_vae(input_shape, latent_dim=latent_dim,
                                        enable_mse=enable_mse, enable_graph=enable_graph)
    vae.compile(optimizer=Adam(0.0002, 0.5), loss=vae_loss)
    pipeline = Pipeline([('preprocessing', scaler), ('vae', vae)])

    mlflow.set_experiment("vae_train")
    with mlflow.start_run():
        mlflow.keras.autolog()
        transformed_X = Pipeline(pipeline.steps[:-1]).fit_transform(X)
        pipeline.fit(transformed_X, transformed_X, vae__batch_size=batch_size, vae__epochs=epochs)

        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.sklearn.log_model(pipeline, "model")
