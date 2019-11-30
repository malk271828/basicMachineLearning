import os
from os.path import splitext, basename, exists
import numpy as np
from datetime import datetime
import hashlib

from imutils import face_utils
import cv2
from tqdm import tqdm
from colorama import *

# Machine Learning Libraries
import dlib

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

class landmarksExtractor(featureExtractor):
    """
    Reference
    ----------
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
    """
    DEFAULT_CACHE_PATH = "./cache/"
    DLIB_CENTER_INDEX = 30
    DLIB_UPPERLIP_INDEX = 62
    DLIB_LOWERLIP_INDEX = 66

    def __init__(self,
                 shape_predictor:str,
                 cache_dir:str = DEFAULT_CACHE_PATH,
                 visualize_window:bool = False):
        """
        :param fileName: If this argument is not a string, video stream will be opened.
        """
        super().__init__(cache_dir=cache_dir)
        self.visualize_window = visualize_window

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)

    def _extractFeature(self, fileName, verbose=0):
        if isinstance(fileName, str):
            cap = cv2.VideoCapture(fileName)
        else:
            cap = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        landmarks_list = list()
        idx_frame = 0
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Converting the image to gray scale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)

                # For each detected face, find the landmark.
                for (i, rect) in enumerate(rects):
                    # Make the prediction and transfom it to numpy array
                    landmarks = self.predictor(gray, rect)
                    landmarks = face_utils.shape_to_np(landmarks)
                    landmarks_list.append(landmarks - landmarks[self.DLIB_CENTER_INDEX])

                    # Draw on our image, all the finded cordinate points (x,y)
                    if verbose > 0:
                        for (i, (x, y)) in enumerate(landmarks):
                            if i == self.DLIB_CENTER_INDEX:
                                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                            else:
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if verbose > 1:
                    if self.visualize_window:
                        # Show the image
                        cv2.imshow("Output", frame)
                    else:
                        cv2.imwrite(self.cache_dir + "{0:03}.png".format(idx_frame), frame)
                        idx_frame += 1

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break 
                # Break the loop
            else: 
                break
    
        # When everything done, release the video capture object
        cap.release()
    
        # Closes all the frames
        cv2.destroyAllWindows()

        return landmarks_list

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
                fileListIterator = tqdm(self.filePathList, ascii=True)
            else:
                fileListIterator = self.filePathList
            for filePath in fileListIterator:
                features = self.singleFileExtractor.getX(verbose=verbose)
                if "samples" in locals():
                    samples = np.vstack([samples, features])
                else:
                    samples = features

            if file_squeeze:
                samples = np.reshape(samples, newshape=(-1,) + (np.prod(samples.shape[1:]),))
            np.savez(self.concatCachePath, features=samples, allow_pickle=True)

        return samples