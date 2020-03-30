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
import torch

import librosa.display

from featureExtractor import *
from landmarkExtractor import *
from lombardFileSelector import *
from vae import *

@pytest.fixture
def dataCorpus():
    class _dataCorpus:
        def __init__(self):
            self.fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")

            # The following message indicates version mismatch between code and model.
            # RuntimeError: Unexpected version found while deserializing dlib::shape_predictor.
            # https://stackoverflow.com/questions/49614460/python-unexpected-version-found-while-deserializing-dlibshape-predictor
            self.SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    return _dataCorpus()

def test_dataFetch(dataCorpus):
    """fetch video data and extract raw audio (pcm)
        play raw audio file the following command:
        >>> aplay -f S16_LE -c2 -r22050 soundfile.raw
    """
    from pytube import YouTube

    yt = YouTube("https://youtu.be/"+dataCorpus.fileID)
    if not os.path.exists(dataCorpus.fileID+".mp4"):
        print("video file is not found")
        yt.streams.first().download(".")
        os.rename(yt.title+".mp4", dataCorpus.fileID+".mp4")

    subprocess.call(["ffmpeg -i {0}.mp4 -vn -f s16le -acodec pcm_s16le soundfile.raw".format(dataCorpus.fileID)], shell=True, cwd=".")

@pytest.mark.landmark
@pytest.mark.parametrize("modal", ["visual", "audio"])
def test_landmarks(dataCorpus, modal):
    fileSelector = dataCorpus.fileSelector

    # w/o specifying cache path
    le1 = landmarksExtractor(dataCorpus.SHAPE_PREDICTOR_PATH)
    landmarks_list = le1.getXy(fileName=fileSelector.getFileList(modal)[0],
                              verbose=2,
                              modality=modal)
    assert 0 != len(landmarks_list)
    print(getShapeListArray(landmarks_list))

    # specifying cache path
    le2 = landmarksExtractor(dataCorpus.SHAPE_PREDICTOR_PATH, cache_dir="./cache2/", visualize_window=True)
    landmarks_list = le2.getXy(fileName=fileSelector.getFileList(modal)[0],
                              verbose=2,
                              modality=modal)
    assert 0 != len(landmarks_list)
    print(getShapeListArray(landmarks_list))

@pytest.mark.landmark
@pytest.mark.parametrize("useCache", [False, True])
@pytest.mark.parametrize("isFlattened", [False, True])
@pytest.mark.parametrize("isOnehot", [False, False])
@pytest.mark.parametrize("sample_shift", [1, 4])
def test_batch( dataCorpus,
                useCache,
                isFlattened,
                isOnehot,
                sample_shift):
    """
    Caution
    -------
    - This test may delete cache directory.
    - The information of sample_shift is not stored into cache.
    """
    fileSelector = dataCorpus.fileSelector
    fextractor = landmarksExtractor(dataCorpus.SHAPE_PREDICTOR_PATH)
    window_size = fextractor.getDim("audio")
    be = batchExtractor(fextractor,
                        window_size=window_size,
                        sample_shift=sample_shift)
    recipe = {
        "visual": fileSelector.getFileList("visual")[:3],
        "audio": fileSelector.getFileList("audio")[:3],
    }
    Xy = be.getXy(recipe=recipe,
                 useCache=useCache,
                 isFlattened=isFlattened,
                 isOnehot=isOnehot,
                 verbose=2)

    for modality in recipe.keys():
        # check if value indexed at center of face is 0
        print("modality:{0} Xy.shape:{1}".format(modality, Xy[modality].shape))
        if isFlattened:
            assert Xy[modality].shape[1] == fextractor.getDim(modality) * window_size
            assert Xy["visual"][0][landmarksExtractor.DLIB_CENTER_INDEX*2 + 0] == 0
            assert Xy["visual"][0][landmarksExtractor.DLIB_CENTER_INDEX*2 + 1] == 0
        else:
            assert Xy[modality].shape[1] == window_size
            assert Xy["visual"][0][0][landmarksExtractor.DLIB_CENTER_INDEX][0] == 0
            assert Xy["visual"][0][0][landmarksExtractor.DLIB_CENTER_INDEX][1] == 0
    #     print("modal:{0} shape:{1}".format(modality, Xy[modality].shape))
    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(2, 1, 1)
    #     librosa.display.specshow(Xy["visual"][:, :, 1].T*5, x_axis="time")
    #     plt.title('visual')
    #     plt.colorbar()
    #     plt.subplot(2, 1, 2)
    #     librosa.display.specshow(Xy["audio"].T, x_axis="time")
    #     plt.title('audio')
    #     plt.tight_layout()
    #     plt.colorbar()
    #     plt.savefig("spec.png")

@pytest.mark.parametrize("useCache", [True])
@pytest.mark.parametrize("batch_size", [10])
def test_pytorch_dataloader(dataCorpus,
                            useCache,
                            batch_size):
    """
    Unifying test between pytorch dataloader and featureExtractor family.
    """
    class lombardDataSet(torch.utils.data.Dataset):
        def __init__(self):
            self.fileSelector = dataCorpus.fileSelector
            self.fextractor = landmarksExtractor(dataCorpus.SHAPE_PREDICTOR_PATH)
            self.be = batchExtractor(self.fextractor,
                                window_size=self.fextractor.getDim("audio"),
                                sample_shift=4)
            self.recipe = {
                "visual": self.fileSelector.getFileList("visual")[:3],
                "audio": self.fileSelector.getFileList("audio")[:3],
            }
            cache_dict = self.be.getCachePathList(recipe=self.recipe)
            self.Xy = self.be.getXy(recipe=self.recipe,
                                    useCache=useCache,
                                    isFlattened=False,
                                    sample_shift=12,
                                    isOnehot=False,
                                    verbose=1)
            self.modalList = list(self.Xy.keys())
            self.num_file = len(cache_dict[self.modalList[0]])
            self.len = len(self.Xy[self.modalList[0]])
        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return {modality:self.Xy[modality][idx] for modality in self.Xy.keys()}

        def getModalList(self):
            return self.modalList

    dataset = lombardDataSet()
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    modalList = dataset.getModalList()
    for batch in train_loader:
        assert batch[modalList[0]].shape[0] == batch_size

def test_lombardFileSelector():
    fileSelector = lombardFileSelector(base_dir="../media/lombardgrid/")

    fileSelector.getFileList("audio", verbose=1)
    fileSelector.getFileList("visual", verbose=1)

def test_mouth_open(dataCorpus):
    fileSelector = dataCorpus.fileSelector
    fextractor = landmarksExtractor(dataCorpus.SHAPE_PREDICTOR_PATH)
    be = batchExtractor(fextractor,
                        window_size=fextractor.getDim("audio"),
                        sample_shift=4)
    recipe = {
        "visual": fileSelector.getFileList("visual")[:3],
        "audio": fileSelector.getFileList("audio")[:3],
    }
    cache_dict = be.getCachePathList(recipe=recipe)
    Xy = be.getXy(recipe=recipe,
                  useCache=True,
                  isFlattened=True,
                  sample_shift=12,
                  isOnehot=False,
                  verbose=1)

    upperLipY = pd.DataFrame([landmarks[landmarksExtractor.DLIB_UPPERLIP_INDEX*2+1] for landmarks in Xy["visual"]])
    lowerLipY = pd.DataFrame([landmarks[landmarksExtractor.DLIB_LOWERLIP_INDEX*2+1] for landmarks in Xy["visual"]])
    mouthCornerR = pd.DataFrame([landmarks[landmarksExtractor.DLIB_MOUTH_CORNER_RIGHT*2] for landmarks in Xy["visual"]])
    mouthCornerL = pd.DataFrame([landmarks[landmarksExtractor.DLIB_MOUTH_CORNER_lEFT*2] for landmarks in Xy["visual"]])

    df = pd.concat([mouthCornerR, mouthCornerL, mouthCornerL - mouthCornerR], axis=1)
    df.columns=["mouthCornerR", "mouthCornerL", "mouthWidth"]
    df.plot()
    plt.show()

    df = pd.concat([upperLipY, lowerLipY, lowerLipY - upperLipY], axis=1)
    df.columns=["upperLipY", "lowerLipY", "MouthOpenLength"]
    df.plot()
    plt.show()
