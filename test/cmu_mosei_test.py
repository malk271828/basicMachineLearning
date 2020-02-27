
import os
import sys
import warnings
import argparse
CMUROOT = "/Users/tsuchiya/CMU-MultimodalSDK/"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, CMUROOT)

from mmsdk import mmdatasdk

def test_load():
    """
    Reference
    ---------
    https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials
    """
    try:
        dataset = mmdatasdk.mmdataset(recipe=mmdatasdk.cmu_mosi.labels,
                                      destination="./cmumosi/")
    except RuntimeError:
        print("already downloaded")

    visual_field = 'CMU_MOSI_VisualFacet_4.1'
    acoustic_field = 'CMU_MOSI_COVAREP'
    text_field = 'CMU_MOSI_ModifiedTimestampedWords'

    features = [
        visual_field
    ]

    recipe = {feat: os.path.join("./cmumosi/", feat) + '.csd' for feat in features}
    dataset = mmdatasdk.mmdataset(recipe)

    firstID = list(dataset[visual_field].keys())[0]
    print(dataset[visual_field][firstID]["features"].shape)