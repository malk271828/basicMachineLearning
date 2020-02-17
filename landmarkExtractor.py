import numpy as np
import soundfile as sf
import cv2

# Machine Learning Libraries
import dlib
from imutils import face_utils

from featureExtractor import featureExtractor

def getShapeListArray(list_array):
    return (len(list_array),) + list_array[0].shape

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
    DLIB_MOUTH_CORNER_RIGHT = 48
    DLIB_MOUTH_CORNER_lEFT = 54

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

    def _extractFeature(self,
                        fileName:str,
                        modality:str = "",
                        verbose:int = 0,
                        **kwargs):
        if modality == "visual":
            if isinstance(fileName, str):
                cap = cv2.VideoCapture(fileName)
            else:
                cap = cv2.VideoCapture(0)

            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")

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
                        if "landmarks_frames" in locals():
                            next_frame = np.expand_dims(landmarks - landmarks[self.DLIB_CENTER_INDEX], 0)
                            landmarks_frames = np.concatenate([landmarks_frames, next_frame], axis=0)
                            assert next_frame.shape == firstframe_shape
                        else:
                            landmarks_frames = landmarks - landmarks[self.DLIB_CENTER_INDEX]
                            landmarks_frames = np.expand_dims(landmarks_frames, 0)
                            firstframe_shape = landmarks_frames.shape

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

            return landmarks_frames

        elif modality == "audio":
            data, samplerate = sf.read(fileName)

            return data
        else:
            raise Exception("modality argument must be passed to invoke this method")