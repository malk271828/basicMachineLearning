import os
import shutil
from os.path import splitext, basename, exists
import numpy as np
from datetime import datetime
import hashlib

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
        if isinstance(fileName, str):
            self.cachePath = self.cache_dir + modality + "/" + splitext(basename(fileName))[0] + self.DEFAULT_CACHE_EXT
        else:
            self.cachePath = self.cache_dir + modality + "/" + str(datetime.now()) + self.DEFAULT_CACHE_EXT

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
        np.savez(self.cachePath, features=features_list, allow_pickle=True)

    def clearCache(self):
        if exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def getDim(self):
        pass

    def getXy(self,
             fileName:str,
             modality:str = "",
             verbose:int = 0,
             **kwargs):
        try:
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
                 cache_dir:str = DEFAULT_CACHE_PATH,
                 file_squeeze:bool = False):
        """
        file_squeeze: boolean, optional
            If enabled, all the features of selected files will be concatenated.
            This flag is NOT saved into cache file, thus client codes have to 
            manage whether loaded cache data have file dimension by your own.
        """
        super().__init__(cache_dir)
        self.singleFileExtractor = singleFileExtractor
        self.file_squeeze = file_squeeze

    def getXy(self,
             filePathList:list,
             modality:str = "",
             verbose:int = 0,
             **kwargs):
        concatPath = "".join(filePathList)
        self.concatCachePath = self.singleFileExtractor.cache_dir + hashlib.md5(concatPath.encode()).hexdigest() + ".npz"
        self.filePathList = filePathList

        return super().getXy(fileName=self.concatCachePath,
                            modality=modality,
                            **kwargs)

    def _extractFeature(self,
                        fileName:str,
                        modality:str = "",
                        verbose:int = 0,
                        **kwargs):
        # extract feature from each file
        if verbose > 0:
            fileListIterator = tqdm(filePathList, ascii=True)
        else:
            fileListIterator = self.filePathList
        samples = list()
        for filePath in fileListIterator:
            features = self.singleFileExtractor.getXy(fileName=filePath,
                                                     modality=modality,
                                                     verbose=verbose)
            samples.append(features)

        if self.file_squeeze:
            for sample in samples:
                if "return_samples" in locals():
                    return_samples = np.concatenate([return_samples, sample], axis=0)
                else:
                    return_samples = sample
                    print(sample.shape)
            samples = return_samples
        if "return_samples" in locals():
            del return_samples
        return samples