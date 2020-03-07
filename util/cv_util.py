import sys
import math

from operator import itemgetter, attrgetter
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from colorama import *
init()

# Image Processing
from skimage.draw import rectangle, rectangle_perimeter
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler

class groupedMinMaxScaler:
    """
    Grouped MinMaxScaler

    Reference
    ---------
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    def __init__(self, grouped_dim: int = 1):
        """
        Constructor

        grouped_dim: int, optional
            specify axes used for grouping with left dimension normalized by MinMaxScaling.
            To get more detail, see method description of computeScaler().
        """
        self.scaler = None
        self.data_max = -sys.maxsize
        self.grouped_dim = grouped_dim

    def _fitScaler(self, array) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(np.reshape(array, newshape=(-1, 1)))
        if self.data_max < scaler.data_max_[0]:
            self.data_max = scaler.data_max_[0]
            self.scaler = scaler

    def computeScaler(self,
                      GroupedArray,
                      verbose: int = 0) -> MinMaxScaler:
        """
        compute scaling factor without applying normalization.
        Groupding axes are specified by grouped_dim.
        e.g. If grouped_dim = 2, looping iterator will traverse for [:,:,...]

        GroupedArray: array, required
            Source array which to be normalized
        """
        for indices in np.ndindex(GroupedArray.shape[:self.grouped_dim]):
            self._fitScaler(GroupedArray[indices])

        if verbose > 0:
            print("data_max:{0}".format(self.data_max))
        assert self.scaler != None
        return self.scaler

    def ApplyScaling(self,
                     GroupedArray,
                     newshape: tuple,
                     verbose: int = 0) -> np.array:
        """
        Apply normalization with computed scaling factor
        """
        if self.scaler == None:
            raise Exception("before invoking this method")
        normalizedArray = GroupedArray.copy()
        for indices in np.ndindex(normalizedArray.shape[:self.grouped_dim]):
            normalized_flattened_array = self.scaler.transform(np.reshape(normalizedArray[indices], newshape=(-1, 1)))
            if newshape == None:
                length = int(math.sqrt(len(normalized_flattened_array)))
                newshape = (length, length)
            normalizedArray[indices] = np.reshape(normalized_flattened_array, newshape=newshape)
        if verbose > 0:
            print("range [{0}, {1}]->[{2}, {3}]".format(np.min(GroupedArray), np.max(GroupedArray), np.min(normalizedArray), np.max(normalizedArray)))
        return normalizedArray

def generateNormalizedPatchedImage(list_grouped_patch_xy:list,
                                   shape:tuple,
                                   mode:str,
                                   grouped_dim:int = 1,
                                   cmStr:str = "jet",
                                   verbose:int = 0,
                                   n_jobs:int = 4) -> tuple:
    """
    Return
    ------
    Tuple of (overlayed image, normalized image, colored image)
    """
    # note:
    # To access shared variables from an inter function to be paralleled,
    # it should be declared as list or numpy array
    list_original_arrays = [0] * len(list_grouped_patch_xy)
    cm = plt.get_cmap(cmStr)

    def _processGroup(i, group):
        original_array_per_layer = np.zeros((len(group),) + (shape[1], shape[0]))
        for indices in np.ndindex(group.shape[:grouped_dim - 1]):
            if verbose > 0:
                print("--------------------")
                print("{0} patches in group {1}-{2}:".format(len(group), i, indices))
            original_array = generatePatchedImage(group[indices], shape, mode=mode,
                                                                cmStr=cmStr,
                                                                verbose=verbose)

            original_array_per_layer[indices] = original_array
        list_original_arrays[i] = original_array_per_layer

    if n_jobs == 1:
        [_processGroup(i, group) for i, group in enumerate(list_grouped_patch_xy)]
    else:
        Parallel(n_jobs=n_jobs,
                require="sharedmem",
                verbose=verbose)( [delayed(_processGroup)(i, group) for i, group in enumerate(list_grouped_patch_xy)] )

    if verbose > 0:
        print("list_grouped_patch_xy.shape: {0}".format((len(list_grouped_patch_xy),) + list_grouped_patch_xy[0].shape))
        print("list_original_arrays.shape: {0}".format((len(list_original_arrays),) + list_original_arrays[0].shape))

    # compute scaling factor for normalization
    gn = groupedMinMaxScaler(grouped_dim=grouped_dim - 1)
    if mode == "add":
        [gn.computeScaler(array) for array in list_original_arrays]
    elif mode == "overwrite" or mode == "overwrite_perimeter":
        pass
    else:
        raise ValueError

    # apply scaling
    if mode == "add":
        grouped_normalized_array = [gn.ApplyScaling(array, newshape=(shape[1], shape[0])) for array in list_original_arrays]
    elif mode == "overwrite" or mode == "overwrite_perimeter":
        grouped_normalized_array = list_original_arrays
    else:
        raise ValueError
    colored_array = cm(grouped_normalized_array)

    if verbose > 0:
        print("")
        print("number of groups: {0}".format(len(list_original_arrays)))
        if mode == "add":
            print("data_max: {0}".format(gn.data_max))
        print("[Grouped] range:[{0}, {1}]".format(np.min(colored_array), np.max(colored_array)))

    return list_original_arrays, grouped_normalized_array, colored_array

def generatePatchedImage(patch_info:np.array,
                         shape:tuple,
                         mode:str,
                         cmStr:str = "jet",
                         verbose:int = 0,
                         n_jobs:int = 1):
    """
    Create multiple-patched image from a list of patch information

    Parameters
    ----------
    patch_info : array of (x, y, cx, cy, alpha)
        specify locations of each patch.
        The range of x and cx must be within [0, height], and y and cy [0, width]
    shape : tuple of (width, height)
        indicate size of image
    mode : string specifying mode
        "add"       - add alpha value of each patch and normalize to limit range to [0, 1]
        "overwrite" - overwrite alpha value on filled shapes without normaling
        "overwrite_perimeter" - overwrite alpha value only on perimeter of shapes without normaling
    cmStr : string specifying color-map
        available color-maps will be listed in the following link:
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    verbose : control verbosity level, default=0
        Lv.1 - show statistics on standard output
        Lv.2 - output generated image to file
    n_jobs : number of thread for multiprocessing parallerism, default=1

    Return
    ------
    Array of a multiple-patch overlayed image

    Reference
    ---------
        https://stackoverflow.com/questions/43457308/is-there-any-good-color-map-to-convert-gray-scale-image-to-colorful-ones-using-p
        https://stackoverflow.com/questions/24571492/stacking-line-drawing-for-float-images-in-pillow
    """
    width, height = shape
    original_array = np.zeros(shape=(height, width), dtype=np.float)

    def _processPatch(patch:np.array):
        x, y, cx, cy, alpha = patch
        x, y, cx, cy = int(x), int(y), int(cx), int(cy)
        # clipping
        if x + cx >= height:
            cx = height - x - 1
        if y + cy >= width:
            cy = width - y - 1
        if cx <= 0 or cy <= 0:
            return

        # pointwise addition
        if mode == "add":
            rr, cc = rectangle(start=(x, y), extent=(cx, cy))
            original_array[rr, cc] += alpha
        elif mode == "overwrite":
            rr, cc = rectangle(start=(x, y), extent=(cx, cy))
            original_array[rr, cc] = alpha
        elif mode == "overwrite_perimeter":
            rr, cc = rectangle_perimeter(start=(x, y), extent=(cx, cy))
            original_array[rr, cc] = alpha
        else:
            raise ValueError

    if n_jobs == 1:
        [_processPatch(patch) for patch in patch_info]
    else:
        Parallel(n_jobs=n_jobs,
                backend='multiprocessing',
                verbose=verbose)([delayed(_processPatch)(patch) for patch in sorted(patch_info, key=itemgetter(4))])

    if verbose > 0:
        print(Fore.CYAN)
        print("[Original] shape: {0} range:[{1}, {2}]".format(original_array.shape, np.min(original_array), np.max(original_array)))
        print(Style.RESET_ALL)
        if verbose > 1:
            VIS_DIR = "visualization/"
            print(Fore.MAGENTA + "output to files to {0}".format(VIS_DIR) + Style.RESET_ALL)

            # output original
            Image.fromarray(original_array).convert("L").save(VIS_DIR + 'original.gif', quality=95)

    return original_array