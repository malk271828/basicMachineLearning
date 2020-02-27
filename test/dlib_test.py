import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
import pandas as pd

# visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam
import mlflow
import mlflow.sklearn
import mlflow.keras

from featureExtractor import *
from landmarkExtractor import *
from lombardFileSelector import *
from vae import *

@pytest.fixture
def expFixture():
    class _expFixture:
        def __init__(self):
            self.fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")

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
@pytest.mark.parametrize("modal", ["visual", "audio"])
def test_landmarks(expFixture, modal):
    fileSelector = expFixture.fileSelector

    # w/o specifying cache path
    le1 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH)
    landmarks_list = le1.getXy(fileName=fileSelector.getFileList(modal)[0],
                              verbose=2,
                              modality=modal)
    assert 0 != len(landmarks_list)
    print(getShapeListArray(landmarks_list))

    # specifying cache path
    le2 = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH, cache_dir="./cache2/", visualize_window=True)
    landmarks_list = le2.getXy(fileName=fileSelector.getFileList(modal)[0],
                              verbose=2,
                              modality=modal)
    assert 0 != len(landmarks_list)
    print(getShapeListArray(landmarks_list))

@pytest.mark.landmark
@pytest.mark.parametrize("file_squeeze", [False, True])
def test_batch( expFixture,
                file_squeeze):
    """
    Caution
    -------
    This test delete cache directory.
    """
    fileSelector = expFixture.fileSelector

    be = batchExtractor(landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH),
                        file_squeeze=file_squeeze)
    be.clearCache()
    recipe = {
        "visual": fileSelector.getFileList("visual")[:10],
        "audio": fileSelector.getFileList("audio")[:10],    
    }
    X = be.getXy(recipe=recipe,
                 verbose=2)
    #print(getShapeListArray(X))

    if file_squeeze:
        assert X["visual"][0][landmarksExtractor.DLIB_CENTER_INDEX][0] == 0
        assert X["visual"][0][landmarksExtractor.DLIB_CENTER_INDEX][1] == 0
    else:
        assert X["visual"][0][0][landmarksExtractor.DLIB_CENTER_INDEX][0] == 0
        assert X["visual"][0][0][landmarksExtractor.DLIB_CENTER_INDEX][1] == 0

def test_lombardFileSelector():
    fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")

    fileSelector.getFileList("audio", verbose=1)
    fileSelector.getFileList("visual", verbose=1)

from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM
def buildmodel():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(10, 128)))
    model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
    
def test_mouth_open(expFixture):
    model = buildmodel()
    le = landmarksExtractor(expFixture.SHAPE_PREDICTOR_PATH)
    be = batchExtractor(le)
    X = be.getX(filePathList=expFixture.filePath, verbose=2)

    upperLipY = pd.DataFrame([landmarks[le.DLIB_UPPERLIP_INDEX*2+1] for landmarks in X])
    lowerLipY = pd.DataFrame([landmarks[le.DLIB_LOWERLIP_INDEX*2+1] for landmarks in X])
    mouthCornerR = pd.DataFrame([landmarks[le.DLIB_MOUTH_CORNER_RIGHT*2] for landmarks in X])
    mouthCornerL = pd.DataFrame([landmarks[le.DLIB_MOUTH_CORNER_lEFT*2] for landmarks in X])

    df = pd.concat([mouthCornerR, mouthCornerL, mouthCornerL - mouthCornerR], axis=1)
    df.columns=["mouthCornerR", "mouthCornerL", "mouthWidth"]
    df.plot()
    plt.show()

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
