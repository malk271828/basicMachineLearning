import numpy as np

# Image Processing
from PIL import Image, ImageDraw, ImageChops

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler

def generateNormalizedPatchedImage(list_patch_xy:np.array,
                                   width:int,
                                   height:int,
                                   verbose:int = 0):
    original_image = Image.new('L', (width, height), (0))
    scaler = MinMaxScaler(feature_range=(0, 255))

    for (x, y, cx, cy, alpha) in list_patch_xy:
        im = Image.new('L', (width, height), (0))
        draw = ImageDraw.Draw(im)
        draw.rectangle(xy=[(x, y), (x+cx, y+cy)], fill=(alpha))
        original_image = ImageChops.add(original_image, im)

    original_array = np.array(original_image)
    normalized_flatten_array = scaler.fit_transform(np.reshape(original_array, newshape=(-1, 1)))
    normalized_array = np.reshape(normalized_flatten_array, newshape=(height, width))

    if verbose > 0:
        print()
        print("[Original] shape: {0}".format(original_array.shape))
        print("[Normalized] shape: {0}".format(normalized_array.shape))

    normalized_image = Image.fromarray(normalized_array)

    original_image.save('original.png', quality=95)
    normalized_image.save('normalized.gif', quality=95)
