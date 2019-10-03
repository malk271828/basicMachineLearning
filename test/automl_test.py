import os
import sys
import warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Machine Learning Libraries
import torch
import autokeras as ak
from keras.datasets import cifar10

def test_automl():
    """
    Reference
    ----------
        https://www.pyimagesearch.com/2019/01/07/auto-keras-and-automl-a-getting-started-guide/
        https://www.simonwenkel.com/2018/09/02/autokeras-cifar10_100.html
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    clf = ak.ImageClassifier(verbose=True)
    clf.fit(X_train, y_train, time_limit=10 * 60 * 60)
    clf.final_fit(X_train, y_train, X_test, y_test, retrain=True)
    print(clf.evaluate(X_test, y_test))
