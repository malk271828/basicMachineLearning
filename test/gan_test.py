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
import pandas as pd

from CGAN import *

@pytest.fixture
def skDataset():
    class _skDataset(torch.utils.data.Dataset):
        def __init__(self,
                     dataset:str,
                     isFlattened=True):
            """
            Parameters
            ----------
                dataset: str, required
                    specify dataset name choosing from "wine"/"cancer"
            """
            if dataset == "cancer":
                # 1d binary classification
                data = load_breast_cancer()
                max_feature_dim = 1
            elif dataset == "wine":
                # 1d multiclass classification
                data = load_wine()
                max_feature_dim = 1
            elif dataset == "digits":
                # 2d multiclass classification
                data = load_digits()
                max_feature_dim = 2
            else:
                raise Exception("Unknown dataset name:{0}".format(dataset))
            self.X, self.y = data["data"], data["target"]
            self.num_class = len(data["target_names"])
            assert len(self.X) == len(self.y)

            self.X = self.X.astype(np.float32)
            self.y = self.y.astype(np.float32)
            hist, bin = np.histogram(self.y, bins=[i for i in range(self.num_class + 1)])

            # reshape data
            self.y = np.expand_dims(self.y, 1)
            if not isFlattened and max_feature_dim != 1:
                self.X = np.reshape(self.X, newshape=(len(self.X), 1, 8, 8))
            self.feature_shape = self.X[0].shape

            print(Fore.CYAN + "\n--------------------------------------")
            print("skDataset ({0}):".format(dataset))
            print("%d samples" % len(self.X))
            print("X shape: {0}".format(self.feature_shape))
            print("y shape: {0}".format(self.y[0].shape))
            print("class histogram:{0}".format(hist))
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
def test_discriminator_1d(skDataset,
                          framework,
                          dataset_name):
    """1d dataset classification test

    Reference:
        https://www.programcreek.com/python/example/104690/sklearn.datasets.load_breast_cancer
    """
    dataset = skDataset(dataset_name)
    batch_size = 100
    num_epochs = 300

    ganfactory = CGANFactory(input_shape=dataset.getFeatureShape(),
                             output_shape=dataset.getFeatureShape(),
                             num_class=1 if dataset.num_class == 2 else dataset.num_class,
                             hidden_size=10,
                             learningRate=0.0001,
                             optimBetas=(0.9, 0.999), # unused
                             batchSize=batch_size,
                             timeSteps=20, # unused
                             generator_type="linear",
                             discriminator_type="linear")

    g, d, trainer = ganfactory.createProductFamily(framework)
    print(g)
    print(d)

    # train discriminator
    if framework == "keras":
        X, y = dataset.getXy()
        if dataset_name == "cancer":
            d.compile(loss=["binary_crossentropy"], optimizer=Adam(0.0002, 0.5))
        elif dataset_name == "wine":
            d.compile(loss=["sparse_categorical_crossentropy"], optimizer=Adam(0.0002, 0.5))
        else:
            raise Exception("unknown dataset name:{0}".format(dataset_name))
        d.fit(X, y, batch_size=batch_size, epochs=num_epochs)

        predicted_y = np.argmax(d.predict(X), axis=1)
        print(classification_report(y, predicted_y))
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        d.train()
        torch.set_num_threads(4)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(d.parameters(), 0.0005)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_corrects = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred_y = d(X)
                    loss = criterion(pred_y, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(pred_y == y)

            epoch_acc = epoch_corrects.double() / len(dataset)

            if epoch % 10 == 0:
                print("Epoch [{0}/{1}] Loss:{2:.4f} Acc:{3:.4f}".format(epoch, num_epochs, epoch_loss, epoch_acc))

@pytest.mark.parametrize("framework", ["pytorch"])
@pytest.mark.parametrize("dataset_name", ["digits"])
def test_discriminator_2d(skDataset,
                          framework,
                          dataset_name):
    """2d dataset classification test

    Reference:
        Basic sample for multi classification task on pytorch:
        https://github.com/lschmiddey/PyTorch-Multiclass-Classification/blob/master/Softmax_Regression_Deep_Learning_Iris_Dataset.ipynb
        set shape of each tensor for pytorch
        https://discuss.pytorch.org/t/runtimeerror-given-groups-1-weight-64-3-3-3-so-expected-input-16-64-256-256-to-have-3-channels-but-got-64-channels-instead/12765
    """
    dataset = skDataset(dataset_name, isFlattened=False)
    batch_size = 100
    num_epochs = 300

    ganfactory = CGANFactory(input_shape=dataset.getFeatureShape(),
                             output_shape=dataset.getFeatureShape(),
                             num_class=1 if dataset.num_class == 2 else dataset.num_class,
                             hidden_size=8,
                             learningRate=0.0001,
                             optimBetas=(0.9, 0.999), # unused
                             batchSize=batch_size,
                             timeSteps=20, # unused
                             generator_type="2dcnn",
                             discriminator_type="2dcnn")

    g, d, trainer = ganfactory.createProductFamily(framework)
    print(g)
    print(d)

    # train discriminator
    if framework == "keras":
        X, y = dataset.getXy()
        if dataset_name == "digits":
            d.compile(loss=["sparse_categorical_crossentropy"], optimizer=Adam(0.0002, 0.5))
        else:
            raise Exception("unknown dataset name:{0}".format(dataset_name))
        d.fit(X, y, batch_size=batch_size, epochs=num_epochs)

        predicted_y = np.argmax(d.predict(X), axis=1)
        print(classification_report(y, predicted_y))
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        d.train()
        #torch.set_num_threads(4)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()
        optimizer = optim.Adam(d.parameters(), 0.0005)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_corrects = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred_y = d(X)
                    print("pred_y", pred_y.shape)
                    print("y", y.shape)
                    loss = criterion(pred_y, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(pred_y == y)

            epoch_acc = epoch_corrects.double() / len(dataset)

            if epoch % 10 == 0:
                print("Epoch [{0}/{1}] Loss:{2:.4f} Acc:{3:.4f}".format(epoch, num_epochs, epoch_loss, epoch_acc))

@pytest.mark.parametrize("framework", ["pytorch"])
@pytest.mark.parametrize("dataset_name", ["cancer"])
def test_adversarial_train_1d(skDataset,
                              framework,
                              dataset_name):
    dataset = skDataset(dataset_name)
    X, y = dataset.getXy()
    batch_size = 100
    num_epochs = 300

    ganfactory = CGANFactory(input_shape=dataset.getFeatureShape(),
                             output_shape=dataset.getFeatureShape(),
                             num_class=2,
                             hidden_size=30,
                             learningRate=0.0001,
                             optimBetas=(0.9, 0.999), # unused
                             batchSize=batch_size,
                             timeSteps=20, # unused
                             generator_type="linear",
                             discriminator_type="linear")

    g, d, trainer = ganfactory.createProductFamily(framework)
    print(g)
    print(d)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    trainer.train_model(generator=g,
                        discriminator=d,
                        dataloader=train_loader)
