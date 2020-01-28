# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_inference.ipynb
# keras <= 2.1.3
# tensorflol <= 1.13.0

import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../ssd_keras")
sys.path.insert(0, "../keras-gradcam")
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

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from grad_cam import *

# original header
from cv_util import *
from lombardFileSelector import *

@pytest.mark.parametrize("target_layer_names", [["input_1", "conv4_3_norm_mbox_conf_reshape", "fc7_mbox_conf_reshape",
                        "conv6_2_mbox_conf_reshape", "conv7_2_mbox_conf_reshape", "conv8_2_mbox_conf_reshape", "conv9_2_mbox_conf_reshape"]])
@pytest.mark.parametrize("entry", [0, 1])
@pytest.mark.parametrize("target_layer", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("mode", ["add", "overwrite", "overwrite_perimeter"])
def test_inference(kerasSSD, entry, target_layer_names, target_layer, mode):
    verbose = 1

    model = kerasSSD
    hidden_layer_models = Model(inputs=model.input, outputs=model.get_layer(target_layer_names[0]).output)
    layer_shape = [model.get_layer(layer_name).input_shape[1:3] for layer_name in target_layer_names]
    outDBoxNums = [model.get_layer(layer_name).output_shape[1] for layer_name in target_layer_names[1:]]
    boxIndexPair = [(0, sum(outDBoxNums))] + [(sum(outDBoxNums[:i]), sum(outDBoxNums[:i+1])) for i in range(len(outDBoxNums))]

    if verbose > 0:
        print(Fore.GREEN + "target layer name: {0}".format(target_layer_names[target_layer]))
        print("layer_shape: {0}".format(layer_shape))
        print("boxIndexPair: {0}".format(boxIndexPair))
        print("boxIndexPair[target_layer]: {0}".format(boxIndexPair[target_layer]) + Style.RESET_ALL)

    #--------------------------------------------------------------------------
    # load test images
    #--------------------------------------------------------------------------
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # We'll only load one image in this example.
    IMG_DIR = "examples/"
    img_paths = fileSelector(IMG_DIR).getFileList()
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

    for img_path in img_paths:
        orig_images.append(imread(img_path))
    img = image.load_img(img_paths[entry], target_size=(model.get_layer("input_1").input_shape[1:3]))
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
        "confidence_thresh" : 0.0000,
        "iou_threshold" : 1.000,
        "top_k" : 8732*len(classes),
        "img_height" : img.shape[0],
        "img_width" : img.shape[1]
    }
    y_pred_original = np.array(decode_detections(y_pred_encoded[0], **decode_param_original))
    list_y_pred = list(map(lambda y: np.array(decode_detections(y[:,slice(*boxIndexPair[target_layer]),:], **decode_param)), y_pred_encoded))

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
    modelPlotPath = VIS_DIR + os.path.basename(model_path) + ".png"
    if not os.path.exists(modelPlotPath):
        plot_model(model, to_file=modelPlotPath, show_shapes=True, show_layer_names=True)

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
            for box in y_pred_thresh[0]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, layer_shape[0][0], layer_shape[0][1])
                if box[0] == target:
                    list_target_patch.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1]))
            list_patch.append(list_target_patch)

            # create predicted result
            for box in y_pred_original_thresh[0]:
                xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, layer_shape[0][0], layer_shape[0][1])
                if box[0] == target:
                    list_predicted_box.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1], target))

        _, _, list_grouped_colored_array, scaler = generateNormalizedGroupedPatchedImage(list_patch,
                                                                                    shape=(orig_image.shape[1], orig_image.shape[0]),
                                                                                    mode=mode,
                                                                                    verbose=verbose)

        for target, colored_array in enumerate(list_grouped_colored_array):
            # create output image
            class_name = classes[target]
            colored_array = np.reshape(colored_array[:,:,:orig_image.shape[2]], newshape=orig_image.shape)
            overlayed_array = (colored_array[:,:,:orig_image.shape[2]]*128+orig_image/2.0).astype(np.uint8)
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
            output_dir = VIS_DIR + os.path.splitext(os.path.basename(img_paths[entry]))[0] + "_" + cmStr + "_" + mode + "/" + target_layer_names[target_layer]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("create dir:{0}".format(output_dir))
            overlayed_img.save(output_dir + "/group{0}_{1}_".format(target, class_name)+cmStr+"_overlayed.bmp")

        # output statistics
        try:
            with open(os.path.dirname(output_dir) + "/stat.txt", mode="a") as fd:
                fd.write("{0}:{1}".format(target_layer_names[target_layer], str(scaler.data_max_)))
        except AttributeError:
            pass

def test_grad(kerasSSD, visualization):
    # load component from fixture
    model = kerasSSD

    IMG_DIR = "examples/"
    layer_name = "input_1"
    mode = "gradcam"

    img_path = fileSelector(IMG_DIR).getFileList()[4]
    orig_image = image.img_to_array(image.load_img(img_path, target_size=(300, 300)))
    X = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
    preprocessed_input = np.expand_dims(orig_image, axis=0)

    gradcam = grad_cam(model, preprocessed_input, 15, layer_name)

    # save file
    visualization.createOutDir(img_path=img_path, mode=mode)
    jetcam = visualization.cm(gradcam)
    overlayed_array = (jetcam[:,:,:orig_image.shape[2]]*128+X/2.0).astype(np.uint8)
    overlayed_img = Image.fromarray(overlayed_array)
    overlayed_img.save(visualization.output_dir + "grad_cam.jpg")

