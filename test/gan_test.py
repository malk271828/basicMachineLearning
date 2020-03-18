import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../ssd_keras")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pytest
from colorama import *
init(autoreset=True)

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

from CGAN import *

@pytest.fixture
def cancerDataset():
    class _cancerDataset():
        def __init__(self):
            self.X, self.y = load_breast_cancer(return_X_y=True)
            assert len(self.X) == len(self.y)
            self.feature_shape = self.X[0].shape
            z_dim = 20
            print(Fore.CYAN + "\n--------------------------------------")
            print("%d samples" % len(self.X))
            print("X shape: {0}".format(self.feature_shape))
            print("y shape: {0}".format(self.y[0].shape))
            print(Fore.CYAN + "--------------------------------------")
            z = np.random.uniform(size=len(self.X))
        def getFeatureShape(self):
            return self.feature_shape
        def getXy(self):
            return self.X, self.y
    return _cancerDataset()

def test_generator_1d_binary(cancerDataset):
    """binary classification test

    Reference:
        https://www.programcreek.com/python/example/104690/sklearn.datasets.load_breast_cancer
    """
    ganfactory = CGANFactory(input_shape=30,#cancerDataset.getFeatureShape(),
                             output_shape=30,#cancerDataset.getFeatureShape(),
                             num_class=2,
                             hidden_size=8,
                             learningRate=0.01, # unused
                             optimBetas=(0.9, 0.999), # unused
                             batchSize=100,
                             timeSteps=20, # unused
                             generator_type="linear",
                             discriminator_type="linear")
    g, d, trainer = ganfactory.createProductFamily("pytorch")

    X, y = cancerDataset.getXy()
    # g.compile(loss=["mean_squared_error"], optimizer=Adam(0.0002, 0.5))
    # g.fit(X, y, batch_size=100, epochs=300)
    trainer.train(X=X, y=y, generator=g, discriminator=d)
    
    # predicted_y = g.predict(x=[z, y])

    # predicted_y = [int(i) if i == 0 else 1 for i in predicted_y]
    # print(classification_report(y, predicted_y))

def test_generator_1d_multi():
    # multiclass classification
    X, y = load_wine(return_X_y=True)
    assert len(X) == len(y)
    feature_shape = X[0].shape

    g = build_generator({"input_shape": feature_shape, "num_class": 3, "hidden_size": 8, "output_shape": 1,
                    "latent_dim": 10, "model_type": "dense"}, verbose=1)

    g.compile(loss=['sparse_categorical_crossentropy'], optimizer=Adam(0.0002, 0.5))
    g.fit([X, y], y, batch_size=100, epochs=100)

def test_generator_2d_multi():
    # multiclass classification
    X, y = load_digits(return_X_y=True)
    assert len(X) == len(y)
    feature_shape = X[0].shape
    X = np.reshape(X, (len(X), 8, 8))

    g = build_generator(1, {"input_shape": feature_shape, "num_class": 10, "hidden_size": 8, "output_shape": 1,
                    "latent_dim": 10, "model_type": "dense"})

    g.compile(loss=['sparse_categorical_crossentropy'], optimizer=Adam(0.0002, 0.5))
    g.fit([X, y], y, batch_size=100, epochs=100)

def test_1():
    X, y = load_boston(return_X_y=True)
    X = X.reshape(len(X), 13, 1)
    y = y.reshape(len(y), 1, 1)
    #X = X.reshape(len(X), 8, 8, 1)

    # Switch concrete factory which you want to create.
    # factory = CGANFactory(input_shape=(13, 1), learningRate=0.01, optimBetas=0.1, batchSize=100, timeSteps=1, 
    #                       generator_type="dense", discriminator_type="dense")
    # g, d, trainer = factory.createProductFamily("keras")

    # trainer.fit(X=X, y=y, generator=g, discriminator=d)
