import numpy as np

# Image Processing
from PIL import Image, ImageDraw, ImageChops
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler

def generateNormalizedPatchedImage(list_patch_xy:np.array,
                                   width:int,
                                   height:int,
                                   cmStr:str = "jet",
                                   verbose:int = 0):
    """
    Parameters
    ----------
    verbose : control verbosity level, default=0
        Lv.1 - show statistics on standard output
        Lv.2 - output generated image to file

    Return
    ------
    tuple of array of normalized image and color-map image

    Reference
    ---------
        https://stackoverflow.com/questions/43457308/is-there-any-good-color-map-to-convert-gray-scale-image-to-colorful-ones-using-p
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    VIS_DIR = "visualization/"
    original_image = Image.new('L', (width, height), (0))
    scaler = MinMaxScaler(feature_range=(0, 1))
    cm = plt.get_cmap(cmStr)

    for (x, y, cx, cy, alpha) in list_patch_xy:
        im = Image.new('L', (width, height), (0))
        draw = ImageDraw.Draw(im)
        draw.rectangle(xy=[(x, y), (x+cx, y+cy)], fill=(alpha))
        original_image = ImageChops.add(original_image, im)

    original_array = np.array(original_image)
    normalized_flatten_array = scaler.fit_transform(np.reshape(original_array, newshape=(-1, 1)))
    normalized_array = np.reshape(normalized_flatten_array, newshape=(height, width))
    colored_array = cm(normalized_array)

    if verbose > 0:
        print()
        print("[Original] shape: {0} range:[{1}, {2}]".format(original_array.shape, np.min(original_array), np.max(original_array)))
        print("[Normalized] shape: {0} range:[{1}, {2}]".format(normalized_array.shape, np.min(normalized_array), np.max(normalized_array)))
        print("[Colored] shape: {0} range:[{1}, {2}]".format(colored_array.shape, np.min(colored_array), np.max(colored_array)))
        if verbose > 1:
            normalized_image = Image.fromarray(normalized_array*255)
            colored_image = Image.fromarray((colored_array[:,:,:]*255).astype(np.uint8))

            original_image.save(VIS_DIR + 'original.png', quality=95)
            normalized_image.save(VIS_DIR + 'normalized.gif', quality=95)
            colored_image.save(VIS_DIR + "colored_" + cmStr + ".png", quality=95)
            print("output to files")

    return normalized_array, colored_array