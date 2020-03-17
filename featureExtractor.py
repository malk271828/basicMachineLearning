import os
import sys
import shutil
from os.path import splitext, basename, exists
import numpy as np
from datetime import datetime
import hashlib
import itertools
import scipy.stats as stats

import cv2
from tqdm import tqdm
from colorama import *

class featureExtractor():
    DEFAULT_CACHE_PATH = "./cache/"
    DEFAULT_CACHE_EXT = ".npz"

    def __init__(self,
                 cache_dir:str = DEFAULT_CACHE_PATH):
        self.cache_dir = cache_dir

    def _loadFromCache(self,
                      fileName:str,
                      modality:str = "",
                      verbose:int = 0):
        if not exists(self.cache_dir + modality):
            os.makedirs(self.cache_dir + modality)

        if exists(self.cachePath):
            data = np.load(self.cachePath, allow_pickle=True)
            if verbose > 0:
                print(Fore.CYAN + "cache file has been loaded :{0}".format(self.cachePath))
                print("{0}".format(data["features"].shape) + Style.RESET_ALL)
            return data["features"]
        else:
            raise FileNotFoundError

    def _saveToCache(self,
                    features_list: list,
                    verbose:int = 0):
        try:
            np.savez(self.cachePath, features=features_list, allow_pickle=True)
        except OverflowError as error:
            # Output expected OverflowErrors.
            print(Fore.RED + str(error) + Style.RESET_ALL)
            if exists(self.cachePath):
                os.remove(self.cachePath)

    def clearCache(self):
        if exists(self.cache_dir):
            print(Fore.YELLOW + "Delete {0}".format(self.cache_dir))
            shutil.rmtree(self.cache_dir)

    def getDim(self):
        raise NotImplemented

    def getCachePath(self,
                     fileName:str,
                     modality:str = ""):
        """
        get and set cache file path from file base name and modality
        """
        # set cache path
        if isinstance(fileName, str):
            self.cachePath = self.cache_dir + modality + "/" + splitext(basename(fileName))[0] + self.DEFAULT_CACHE_EXT
        else:
            self.cachePath = self.cache_dir + modality + "/" + str(datetime.now()) + self.DEFAULT_CACHE_EXT

        return self.cachePath

    def getXy(self,
             fileName:str,
             modality:str = "",
             useCache:bool = True,
             verbose:int = 0,
             **kwargs):
        try:
            self.getCachePath(fileName, modality)

            if not useCache:
                raise FileNotFoundError
            if verbose > 0:
                print(Fore.CYAN + "trying to load : {0}".format(fileName) + Style.RESET_ALL)
            features_list = self._loadFromCache(fileName=fileName, modality=modality, verbose=verbose)
        except FileNotFoundError:
            features_list = self._extractFeature(fileName=fileName, modality=modality, verbose=verbose, **kwargs)
            self._saveToCache(features_list=features_list, verbose=verbose)

        return features_list

    def _extractFeature(self,
                        fileName:str,
                        modality:str = "",
                        verbose:int = 0,
                        **kwargs):
        """
        Parameters
        ----------
        fileName: file path to extract feature

        Return
        ------
        Array of a feature extracted from single file
        """
        raise NotImplemented

def padStack(a):
    b = np.zeros([len(a), len(max(a, key = lambda x: len(x)))])
    for i, j in enumerate(a):
        b[i][:len(j)] = j
    return b

class batchExtractor(featureExtractor):
    """
    Decorator pattern batchExtractor

    """
    DEFAULT_CACHE_PATH = "./cache/"
    DEFAULT_CACHE_EXT = ".npz"

    def __init__(self,
                 singleFileExtractor:featureExtractor,
                 window_size:int,
                 cache_dir:str = DEFAULT_CACHE_PATH,
                 sample_shift:int = 0):
        """
        sample_shift: int, optional
            If this argument is positive value, all the features of selected
            files will be sliced at interval of sample_shift. This value is
            NOT saved into cache file, thus client codes have to manage whether
            loaded cache data have file dimension by your own.
        """
        super().__init__(cache_dir)
        self.singleFileExtractor = singleFileExtractor
        self.sample_shift = sample_shift
        self.window_size = self.singleFileExtractor.getDim()

    def getCachePathList(self,
                         recipe:dict) -> dict:
        num_files = len(recipe[list(recipe.keys())[0]])
        for fileIdx in np.arange(num_files):
            for modality in recipe.keys():
                recipe[modality][fileIdx] = super().getCachePath(fileName=recipe[modality][fileIdx],
                                                                 modality=modality)
        return recipe

    def getXy(self,
              recipe:dict(),
              useCache:bool = True,
              verbose:int = 0,
              **kwargs):
        """
        Get feature of multiple modalities

        recipe: dictionary, required
            The keys indicates each modality and the values the list of file pathes to be loaded.S
            For example:
            {
                "visual": list of visual modality source files,
                "audio": list of audio modality source files
            }
        """
        allmodalConcatFile = "".join(list(itertools.chain.from_iterable(recipe.values())))
        concatCachePath = self.singleFileExtractor.cache_dir + hashlib.md5(allmodalConcatFile.encode()).hexdigest() + ".npz"

        feature = super().getXy(fileName=concatCachePath, recipe=recipe, useCache=useCache, verbose=verbose, **kwargs)

        # savez method dictionary as ndarray
        if type(feature)==np.ndarray:
            feature = feature.item()
        return feature

    def _extractFeature(self,
                        verbose:int = 0,
                        **kwargs):
        """
        recipe: dictionary, required
        """
        # check arguments
        recipe = kwargs["recipe"]
        isFlattened = kwargs["isFlattened"]
        isOnehot = kwargs["isOnehot"]

        # extract feature from each file
        self.num_files = len(recipe[list(recipe.keys())[0]])
        if verbose > 0:
            fileIdxIterator = tqdm(np.arange(self.num_files), ascii=True, desc="extracting")
        else:
            fileIdxIterator = np.arange(self.num_files)

        features = dict()
        for fileIdx in fileIdxIterator:
            min_length = sys.maxsize
            for modality in recipe.keys():
                features_per_file = self.singleFileExtractor.getXy(fileName=recipe[modality][fileIdx],
                                                          modality=modality,
                                                          verbose=verbose)
                if modality in features.keys():
                    features[modality].append(features_per_file)
                else:
                    features[modality] = [features_per_file]
                min_length = min(min_length, len(features_per_file))

            # align length of each modality
            for modality in recipe.keys():
                features[modality][fileIdx] = features[modality][fileIdx][:min_length]

        feature_shape = dict()
        if self.sample_shift > 0:
            num_total_sample = 0
            for modality in features.keys():
                for features_per_file in features[modality]:
                    if modality not in feature_shape.keys():
                        feature_shape[modality] = features_per_file[0].shape
                    num_total_sample += int( (len(features_per_file) - self.window_size) / self.sample_shift)

            print("feature_shape: {0}".format(feature_shape))
            for modality in features.keys():
                if verbose > 0:
                    print("sampling... modality:{0}".format(modality))

                if modality == "ref" or modality == "label":
                    samples = np.zeros((num_total_sample, ) + feature_shape[modality])
                else:
                    if isFlattened:
                        samples = np.zeros((num_total_sample, self.window_size * np.prod(feature_shape[modality])))
                    else:
                        samples = np.zeros((num_total_sample, self.window_size) + feature_shape[modality])

                total_current_pos = 0
                file_shift = 0
                for fileIdx, features_per_file in enumerate(features[modality]):
                    num_sample = int( (len(features_per_file) - self.window_size) / self.sample_shift)

                    for sampleIdx in range(num_sample):
                        start = sampleIdx * self.sample_shift
                        end = sampleIdx * self.sample_shift + self.window_size
                        if modality == "ref" or modality == "label":
                            mode_val, mode_num = stats.mode(features_per_file[start:end])
                            sample = mode_val
                        else:
                            if isFlattened:
                                sample = np.array(features_per_file[start:end]).flatten()
                            else:
                                sample = np.array(features_per_file[start:end])
                        samples[file_shift + sampleIdx] = sample
                    total_current_pos += len(features_per_file)
                    file_shift = int((total_current_pos - self.window_size) / self.sample_shift)
                features[modality] = samples
        return features