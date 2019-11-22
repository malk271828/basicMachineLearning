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

DEFAULT_CACHE_PATH = "./cache/"
DLIB_CENTER_INDEX = 30

class landmarksExtractor():
    """
    Reference
    ----------
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
    """
    def __init__(self,
                 shape_predictor:str,
                 fileName,
                 cache_dir:str = DEFAULT_CACHE_PATH,
                 visualize_window:bool = False):
        """
        :param fileName: If this argument is not a string, video stream will be opened.
        """
        self.cache_dir = cache_dir
        self.visualize_window = visualize_window
        if not exists(cache_dir):
            os.makedirs(self.cache_dir)

        if isinstance(fileName, str):
            self.cap = cv2.VideoCapture(fileName)
            self.cachePath = self.cache_dir + splitext(basename(fileName))[0] + ".npz"
        else:
            self.cap = cv2.VideoCapture(0)
            self.cachePath = self.cache_dir + str(datetime.now()) + ".npz"

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)

    def getLandmarks(self, verbose=0):
        # search cache
        if exists(self.cachePath):
            data = np.load(self.cachePath)
            if verbose > 0:
                print(Fore.CYAN + "cache file has been loaded :{0}".format(self.cachePath))
                print("{0}".format(data["landmarks"].shape) + Style.RESET_ALL)
            return data["landmarks"]
        else:
            landmarks_list = list()
            idx_frame = 0
            # Read until video is completed
            while(self.cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if ret:
                    # Converting the image to gray scale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = self.detector(gray, 0)

                    # For each detected face, find the landmark.
                    for (i, rect) in enumerate(rects):
                        # Make the prediction and transfom it to numpy array
                        landmarks = self.predictor(gray, rect)
                        landmarks = face_utils.shape_to_np(landmarks)
                        landmarks_list.append(landmarks - landmarks[DLIB_CENTER_INDEX])

                        # Draw on our image, all the finded cordinate points (x,y)
                        if verbose > 0:
                            for (i, (x, y)) in enumerate(landmarks):
                                if i == DLIB_CENTER_INDEX:
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
            self.cap.release()
        
            # Closes all the frames
            cv2.destroyAllWindows()

            # save extracted landmarks
            np.savez(self.cachePath, landmarks=landmarks_list, allow_pickle=True)

            return landmarks_list

class batchExtractor():
    def __init__(self,
                 shape_predictor:str,
                 filePathList:list,
                 cache_dir:str = DEFAULT_CACHE_PATH,
                 visualize_window:bool = False):
        self.filePathList = filePathList
        self.shape_predictor = shape_predictor
        concatPath = "".join(filePathList)
        self.concatCachePath = cache_dir + hashlib.md5(concatPath.encode()).hexdigest() + ".npz"
        self.cache_dir = cache_dir
        self.visualize_window = visualize_window

    def getX(self,
             file_squeeze:bool = True,
             verbose:int = 0):
        if exists(self.concatCachePath):
            # load serialized feature file
            samples = np.load(self.concatCachePath)["landmarks"]
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
                le = landmarksExtractor(self.shape_predictor, filePath, cache_dir=self.cache_dir,
                                        visualize_window=self.visualize_window)
                landmarks = le.getLandmarks(verbose=verbose)
                if "samples" in locals():
                    samples = np.vstack([samples, landmarks])
                else:
                    samples = landmarks

            if file_squeeze:
                samples = np.reshape(samples, newshape=(-1,) + samples.shape[2:])
            np.savez(self.concatCachePath, landmarks=samples, allow_pickle=True)

        return samples