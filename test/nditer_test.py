import numpy as np

def test_nd():
    # create data
    n = 3
    array = np.arange(0, 16, 1).reshape(2, 2, 2, 2)

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
