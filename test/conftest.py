import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../keras-gradcam")

import pytest
from colorama import *
init()

@pytest.fixture
def visualization():
    # visualization module
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class _visualization():
        def __init__(self):
            self.VIS_DIR = "visualization/"
            self.cmStr = "jet"
            self.cm = plt.get_cmap(self.cmStr)

        def createOutDir(self, **kwargs):
            if "target_layer_name" in kwargs.keys():
                self.output_dir = self.VIS_DIR + os.path.splitext(os.path.basename(kwargs["img_path"]))[0] + "_" + self.cmStr + "_" + kwargs["mode"] + "/" + kwargs["target_layer_name"]
            else:
                self.output_dir = self.VIS_DIR + os.path.splitext(os.path.basename(kwargs["img_path"]))[0] + "_" + self.cmStr + "_" + kwargs["mode"] + "/"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(Fore.CYAN + "create dir:{0}".format(self.output_dir) + Style.RESET_ALL)

    return _visualization()

@pytest.fixture(params=[2, 3])
def kerasSSD(request, scope="module"):
    sys.path.insert(0, "../ssd_keras")

    # Machine Learning module
    from keras import backend as K
    from keras.models import load_model, Model
    from keras.utils import plot_model

    from keras_loss_function.keras_ssd_loss import SSDLoss
    from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
    from keras_layers.keras_layer_DecodeDetections import DecodeDetections
    from keras_layers.keras_layer_L2Normalization import L2Normalization

    #--------------------------------------------------------------------------
    # Load trained model
    #--------------------------------------------------------------------------

    model_path = "ssd300_pascal_07+12_102k_steps.h5"
    VIS_DIR = "visualization/"

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session() # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'DecodeDetections': DecodeDetections,
                                                'compute_loss': ssd_loss.compute_loss})

    modelPlotPath = VIS_DIR + os.path.basename(model_path) + ".png"
    if not os.path.exists(modelPlotPath):
        plot_model(model, to_file=modelPlotPath, show_shapes=True, show_layer_names=True)

    classes = ['background',
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

    return model, classes, request.param
