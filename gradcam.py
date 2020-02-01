from __future__ import print_function

import numpy as np
import sys
import math

import cv2

from keras import backend as K
from keras.models import load_model, Model
import tensorflow as tf

def gradcam(input_model,
             image: np.array,
             cls: int,
             layer_name: str,
             verbose: int = 0,
             pred_y: np.array = None) -> np.array:
    """
    GradCAM method for visualizing input saliency.

    Parameters
    ----------
    image: numpy array shaped with (batch size, width, height, channel)

    cls: interger representing class
    
    layer_name: 

    Reference
    ---------
    https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py
    https://towardsdatascience.com/using-tf-print-in-tensorflow-aa26e1cff11e
    """
    y_c = input_model.output[:, :, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    # Normalize if necessary
    # grads = tf.keras.backend.l2_normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    weights = np.mean(grads_val[0], axis=(0, 1))
    cam = np.dot(output[0], weights)

    # Process CAM
    cam = cv2.resize(cam, image.shape[1:3], cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    # cam_max = cam.max() 
    # if cam_max != 0: 
    #     cam = cam / cam_max

    if verbose > 0:
        print("cls:{0}".format(cls))
        print("shape of y_c:{0}".format(y_c.get_shape()))
        print("shape of conv_output:{0}".format(conv_output.get_shape()))
        print("shape of cam:{0}".format(cam.shape))
        #print("cam_max:{0}".format(cam_max))
        tf.print(y_c, [pred_y], output_stream=sys.stderr)

    return cam