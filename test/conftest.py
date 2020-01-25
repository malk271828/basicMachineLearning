import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../keras-gradcam")
import pytest

from keras import backend as K
from keras.models import load_model, Model

@pytest.fixture
def kerasSSD(scope="module"):
    sys.path.insert(0, "../ssd_keras")
    from keras_loss_function.keras_ssd_loss import SSDLoss
    from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
    from keras_layers.keras_layer_DecodeDetections import DecodeDetections
    from keras_layers.keras_layer_L2Normalization import L2Normalization

    #--------------------------------------------------------------------------
    # Load trained model
    #--------------------------------------------------------------------------

    model_path = "ssd300_pascal_07+12_102k_steps.h5"
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session() # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'DecodeDetections': DecodeDetections,
                                                'compute_loss': ssd_loss.compute_loss})

    return model
