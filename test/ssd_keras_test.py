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

# Image Processing
from imageio import imread
import numpy as np
from skimage.draw import rectangle_perimeter

# Machine Learning 
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

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

# Set the image size.
img_height = 300
img_width = 300


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'ssd300_pascal_07+12_102k_steps.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
K.clear_session() # Clear previous models from memory.
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})


orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
IMG_DIR = "examples/"
img_paths = [IMG_DIR + "fish_bike.jpg",
             IMG_DIR + "cat_and_dog.jpg",
             IMG_DIR + "diningTbl.jpg"]
entry = 0

for img_path in img_paths:
    orig_images.append(imread(img_path))
img = image.load_img(img_paths[entry], target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)
y_pred_original = np.array(decode_detections(y_pred, confidence_thresh=0.1, iou_threshold=0.4, top_k=200,
                                                img_height=img.shape[0], img_width=img.shape[1]))
y_pred = np.array(decode_detections(y_pred, confidence_thresh=0.0001, iou_threshold=0.999, top_k=8732,
                                                img_height=img.shape[0], img_width=img.shape[1]))

confidence_threshold = 0.00
confidence_threshold_original = 0.5

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
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
        list_target_patch = list()

        # create confidence map
        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, img_width, img_height)
            if box[0] == target:
                list_target_patch.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1]))
        list_patch.append(list_target_patch)

        # create predicted result
        for box in y_pred_original_thresh[0]:
            xmin, ymin, xmax, ymax = transformCordinate(box, orig_image, img_width, img_height)
            if box[0] == target:
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                list_predicted_box.append((int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin), box[1], target))

    _, _, list_grouped_colored_array, _ = generateNormalizedGroupedPatchedImage(list_patch, 
                                                                                shape=(orig_image.shape[1], orig_image.shape[0]),
                                                                                verbose=2)

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
        output_dir = VIS_DIR + os.path.splitext(os.path.basename(img_paths[entry]))[0] + "_" + cmStr
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("create dir:{0}".format(output_dir))
        overlayed_img.save(output_dir + "/group{0}_{1}_".format(target, class_name)+cmStr+"_overlayed.bmp")

        # verbose standard output





