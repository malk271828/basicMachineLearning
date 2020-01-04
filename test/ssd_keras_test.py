# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_inference.ipynb
# keras <= 2.1.3
# tensorflol <= 1.13.0

import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../ssd_keras")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import pytest

# Image Processing
from imageio import imread
import numpy as np
from skimage.draw import rectangle_perimeter

# Machine Learning 
from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils import plot_model

# visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# original header
from cv_util import *

@pytest.mark.parametrize("target_layer_names", [["conv4_3_norm_mbox_conf_reshape", "fc7_mbox_conf_reshape",
                        "conv6_2_mbox_conf_reshape", "conv7_2_mbox_conf_reshape", "conv8_2_mbox_conf_reshape", "conv9_2_mbox_conf_reshape"]])
@pytest.mark.parametrize("entry", [2])
@pytest.mark.parametrize("target_layer", [6])
def test_inference(entry, target_layer_names, target_layer):
    verbose = 1
    #--------------------------------------------------------------------------
    # Load trained model
    #--------------------------------------------------------------------------

    # Set the path to the `.h5` file of the model to be loaded.
    model_path = 'ssd300_pascal_07+12_102k_steps.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session() # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'DecodeDetections': DecodeDetections,
                                                'compute_loss': ssd_loss.compute_loss})

    hidden_layer_models = Model(inputs=model.input, outputs=model.get_layer(target_layer_names[0]).output)
    layer_shape = [model.get_layer(layer_name).input_shape[1:3] for layer_name in target_layer_names]
    outDBoxNums = [model.get_layer(layer_name).output_shape[1] for layer_name in target_layer_names]
    boxIndexPair = [(0, sum(outDBoxNums))] + [(sum(outDBoxNums[:i]), sum(outDBoxNums[:i+1])) for i in range(len(outDBoxNums))]
    layer_shape = [(300, 300)] + layer_shape

    if verbose > 0:
        #print("target layer name: {0}".format(target_layer_names[target_layer]))
        print(layer_shape)
        print(boxIndexPair)
        print(boxIndexPair[target_layer])
        
    #--------------------------------------------------------------------------
    # load test images
    #--------------------------------------------------------------------------
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # We'll only load one image in this example.
    IMG_DIR = "examples/"
    img_files = ["fish_bike.jpg",
                "cat_and_dog.jpg",
                "diningTbl.jpg",
                "cow_and_horse.jpg"]

    for img_path in img_files:
        orig_images.append(imread(IMG_DIR + img_path))
    img = image.load_img(IMG_DIR + img_files[entry], target_size=(model.get_layer("input_1").input_shape[1:3]))
    img = image.img_to_array(img) 
    input_images.append(img)
    input_images = np.array(input_images)

    #--------------------------------------------------------------------------
    # Inference & Decode
    #--------------------------------------------------------------------------
    models = [model]
    y_pred_encoded = list(map(lambda m: m.predict(input_images), models))
    decode_param_original = {
        "confidence_thresh" : 0.1,
        "iou_threshold" : 0.4,
        "top_k" : 200,
        "img_height" : img.shape[0],
        "img_width" : img.shape[1]
    }
    decode_param = {
        "confidence_thresh" : 0.0001,
        "iou_threshold" : 0.999,
        "top_k" : 8732,
        "img_height" : img.shape[0],
        "img_width" : img.shape[1]
    }
    y_pred_original = np.array(decode_detections(y_pred_encoded[0], **decode_param_original))
    list_y_pred = list(map(lambda y: np.array(decode_detections(y, **decode_param)), y_pred_encoded))

    #--------------------------------------------------------------------------
    # Filtering
    #--------------------------------------------------------------------------
    confidence_threshold_original = 0.4
    confidence_threshold = 0.00

    y_pred_thresh = [[y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])] for y_pred in list_y_pred][0]
    y_pred_original_thresh = [y_pred_original[k][y_pred_original[k,:,1] > confidence_threshold_original] for k in range(y_pred_original.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted {0} boxes:\n".format(len(y_pred_thresh[0])))
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])


    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    VIS_DIR = "visualization/"
    cmStr = "jet"
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
    plot_model(model, to_file=VIS_DIR + os.path.basename(model_path) + ".png", show_shapes=True, show_layer_names=True)

    def transformCordinate(box, orgImage, img_width, img_height):
        xmin = box[2] * orgImage.shape[1] / img_width
        ymin = box[3] * orgImage.shape[0] / img_height
        xmax = box[4] * orgImage.shape[1] / img_width
        ymax = box[5] * orgImage.shape[0] / img_height

        return xmin, ymin, xmax, ymax

    for idx_image, orig_image in enumerate(orig_images[entry:entry+1]):
        list_patch = list()
        list_predicted_box = list()
        for target, class_name in enumerate(classes):
            # create confidence map
            list_target_patch = list()
            for box in y_pred_thresh[0][boxIndexPair[target_layer][0]:boxIndexPair[target_layer][1]]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, layer_shape[0][0], layer_shape[0][1])
                if box[0] == target:
                    list_target_patch.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1]))
            list_patch.append(list_target_patch)

            # create predicted result
            for box in y_pred_original_thresh[0]:
                xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, layer_shape[0][0], layer_shape[0][1])
                if box[0] == target:
                    color = colors[int(box[0])]
                    list_predicted_box.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1], target))

        _, _, list_grouped_colored_array, scaler = generateNormalizedGroupedPatchedImage(list_patch,
                                                                                    shape=(orig_image.shape[1], orig_image.shape[0]),
                                                                                    verbose=verbose)

        for target, colored_array in enumerate(list_grouped_colored_array):
            # create output image
            class_name = classes[target]
            colored_array = np.reshape(colored_array[:,:,:3], newshape=(orig_image.shape[0], orig_image.shape[1], 3))
            overlayed_array = (colored_array[:,:,:3]*128+orig_image/2.0).astype(np.uint8)
            for predicted_box in list_predicted_box:
                if predicted_box[5] == target:
                    rr, cc = rectangle_perimeter(start=(predicted_box[0], predicted_box[1]),
                                                extent=(predicted_box[2], predicted_box[3]),
                                                shape=(orig_image.shape[0], orig_image.shape[1]))
                    overlayed_array[rr, cc] = 255
            overlayed_img = Image.fromarray(overlayed_array)
            for predicted_box in list_predicted_box:
                if target == predicted_box[5]:
                    ImageDraw.Draw(overlayed_img).text((predicted_box[1], predicted_box[0]), "{0}:{1:.3g}".format(class_name, predicted_box[4]))

            # create output path and directory
            output_dir = VIS_DIR + os.path.splitext(os.path.basename(img_files[entry]))[0] + "_" + cmStr
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("create dir:{0}".format(output_dir))
            overlayed_img.save(output_dir + "/group{0}_{1}_".format(target, class_name)+cmStr+"_overlayed.bmp")

            # verbose standard output
            
