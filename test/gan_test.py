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

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.metrics import classification_report

from CGAN import *

@pytest.fixture
def skDataset():
    class _skDataset(torch.utils.data.Dataset):
        def __init__(self, dataset:str):
            """
            Parameters
            ----------
                dataset: str, required
                    specify dataset name choosing from "wine"/"cancer"
            """
            if dataset == "cancer":
                # 1d binary classification
                self.X, self.y = load_breast_cancer(return_X_y=True)
            elif dataset == "wine":
                # 1d multiclass classification
                self.X, self.y = load_wine(return_X_y=True)
            elif dataset == "digit":
                # 2d multiclass classification
                self.X, self.y = load_digit(return_X_y=True)
            else:
                raise Exception("Unknown dataset name:{0}".format(dataset))
            assert len(self.X) == len(self.y)
            self.feature_shape = self.X[0].shape

            self.X = self.X.astype(np.float32)
            self.y = self.y.astype(np.float32)
            self.y = np.expand_dims(self.y, 1)

            print(Fore.CYAN + "\n--------------------------------------")
            print("skDataset ({0}):".format(dataset))
            print("%d samples" % len(self.X))
            print("X shape: {0}".format(self.feature_shape))
            print(self.X)
            print("y shape: {0}".format(self.y[0].shape))
            print(self.y)
            print(Fore.CYAN + "--------------------------------------")
            z = np.random.uniform(size=len(self.X))
        def getFeatureShape(self):
            return self.feature_shape
        def getXy(self):
            return self.X, self.y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    return _skDataset

@pytest.mark.parametrize("framework", ["pytorch"])
@pytest.mark.parametrize("dataset_name", ["cancer"])
def test_generator_1d(skDataset,
                      framework,
                      dataset_name):
    """binary classification test

    Reference:
        https://www.programcreek.com/python/example/104690/sklearn.datasets.load_breast_cancer
    """
    dataset = skDataset(dataset_name)
    X, y = dataset.getXy()
    batch_size = 100
    num_epochs = 300

    ganfactory = CGANFactory(input_shape=dataset.getFeatureShape(),
                             output_shape=1,#skDataset.getFeatureShape(),
                             num_class=2,
                             hidden_size=12,
                             learningRate=0.0001,
                             optimBetas=(0.9, 0.999), # unused
                             batchSize=batch_size,
                             timeSteps=20, # unused
                             generator_type="linear",
                             discriminator_type="linear")

    g, d, trainer = ganfactory.createProductFamily(framework)
    print(g)
    print(d)

    # train
    if framework == "keras":
        if dataset_name == "cancer":
            g.compile(loss=["binary_crossentropy"], optimizer=Adam(0.0002, 0.5))
        elif dataset_name == "wine":
            g.compile(loss=["sparse_categorical_crossentropy"], optimizer=Adam(0.0002, 0.5))
        else:
            raise Exception("unknown dataset name:{0}".format(dataset_name))
        g.fit(X, y, batch_size=batch_size, epochs=num_epochs)

        predicted_y = g.predict(X)
        predicted_y = [int(i) if i == 0 else 1 for i in predicted_y]
        print(classification_report(y, predicted_y))
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True
        )
        g.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(g.parameters(), 0.001)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_corrects = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred_y = g(X)
                    loss = criterion(pred_y, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(pred_y == y)

            epoch_acc = epoch_corrects.double() / len(dataset)

            if epoch % 10 == 0:
                print("Epoch [{0}/{1}] Loss:{2:.4f} Acc:{3:.4f}".format(epoch, num_epochs, epoch_loss, epoch_acc))

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
