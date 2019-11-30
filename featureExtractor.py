import os
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
        if not exists(cache_dir):
            os.makedirs(self.cache_dir)

    def loadFromCache(self,
                      fileName:str,
                      verbose:int = 0):
        if isinstance(fileName, str):
            self.cachePath = self.cache_dir + splitext(basename(fileName))[0] + self.DEFAULT_CACHE_EXT
        else:
            self.cachePath = self.cache_dir + str(datetime.now()) + self.DEFAULT_CACHE_EXT

        if exists(self.cachePath):
            data = np.load(self.cachePath, allow_pickle=True)
            if verbose > 0:
                print(Fore.CYAN + "cache file has been loaded :{0}".format(self.cachePath))
                print("{0}".format(data["features"].shape) + Style.RESET_ALL)
            return data["features"]
        else:
            raise FileNotFoundError

    def saveToCache(self,
                    features_list: list,
                    verbose:int = 0):
        np.savez(self.cachePath, features=features_list, allow_pickle=True)

    def getX(self,
             fileName:str,
             verbose:int = 0):
        try:
            features_list = self.loadFromCache(fileName=fileName, verbose=verbose)
        except FileNotFoundError:
            features_list = self._extractFeature(fileName=fileName)
            self.saveToCache(features_list=features_list, verbose=verbose)

        return features_list

    def _extractFeature(self,
                        fileName:str,
                        verbose:int = 0):
        raise NotImplemented

class batchExtractor():
    DEFAULT_CACHE_PATH = "./cache/"

    def __init__(self,
                 singleFileExtractor:featureExtractor):
        self.singleFileExtractor = singleFileExtractor

    def getX(self,
             filePathList:list,
             file_squeeze:bool = True,
             verbose:int = 0):
        concatPath = "".join(filePathList)
        self.concatCachePath = self.singleFileExtractor.cache_dir + hashlib.md5(concatPath.encode()).hexdigest() + ".npz"

        if exists(self.concatCachePath):
            # load serialized feature file
            samples = np.load(self.concatCachePath)["features"]
            if verbose > 0:
                print(Fore.CYAN + "concat cache file has been loaded :{0}".format(self.concatCachePath))
                print("{0}".format(samples.shape) + Style.RESET_ALL)
        else:
            # extract feature from each file
            if verbose > 0:
                fileListIterator = tqdm(filePathList, ascii=True)
            else:
                fileListIterator = self.filePathList
            for filePath in fileListIterator:
                features = self.singleFileExtractor.getX(fileName=filePath, verbose=verbose)
                if "samples" in locals():
                    samples = np.vstack([samples, features])
                else:
                    samples = features

            if file_squeeze:
                samples = np.reshape(samples, newshape=(-1,) + (np.prod(samples.shape[1:]),))
            np.savez(self.concatCachePath, features=samples, allow_pickle=True)

        return samples