import numpy as np
import pytest

from util.cv_util import *

@pytest.fixture(params=[1, 2, 3])
def createArray(request):
    # create data
    n = request.param
    array = np.arange(0, 16, 1).reshape(2, 2, 2, 2)

    return array, n

def test_nditer(createArray):
    array, n = createArray

    # nditer test
    it = np.nditer( array[0],
                    flags=["multi_index", "reduce_ok"],
                    op_flags=['readwrite'],
                    itershape=(2,)*n)
    while not it.finished:
        print("index:{0}, value:{1}".format(it.multi_index, it[0]))

        it.iternext()

    # Ellipsis test
    for subarray in array[..., :]:
        print(subarray.shape)

def test_ndindex(createArray):
    """
    numpy ndindex test
    """
    array, n = createArray
    for indices in np.ndindex(array.shape[:n]):
        print("indices:{0} value:{1}".format(indices, array[indices]))

def test_grouped_minmaxScaler(createArray):
    array, n = createArray
    gn = groupedNorm()
    gn.computeScaler(array, grouped_dim=n)
