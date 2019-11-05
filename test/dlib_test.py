import os
import sys
import warnings
import subprocess
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from imutils import face_utils
import pytest
import cv2

# Machine Learning Libraries
import dlib

@pytest.fixture
def expFixture():
    class _expFixture:
        def __init__(self):
            self.fileID = "eIu0CXKekI4"

            # The following message indicates version mismatch between code and model.
            # RuntimeError: Unexpected version found while deserializing dlib::shape_predictor.
            # https://stackoverflow.com/questions/49614460/python-unexpected-version-found-while-deserializing-dlibshape-predictor
            self.SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

    return _expFixture()

def test_dataFetch(expFixture):
    """fetch video data and extract raw audio (pcm)
        play raw audio file the following command:
        >>> aplay -f S16_LE -c2 -r22050 soundfile.raw
    """
    from pytube import YouTube
    expFixture.fileID = "eIu0CXKekI4"

    yt = YouTube("https://youtu.be/"+expFixture.fileID)
    if not os.path.exists(expFixture.fileID+".mp4"):
        print("video file is not found")
        yt.streams.first().download(".")
        os.rename(yt.title+".mp4", expFixture.fileID+".mp4")

    subprocess.call(["ffmpeg -i {0}.mp4 -vn -f s16le -acodec pcm_s16le soundfile.raw".format(expFixture.fileID)], shell=True, cwd=".")

def test_load(expFixture):
    """
    Reference
    ----------
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
    """
    cap = cv2.VideoCapture(expFixture.fileID+".mp4")

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(expFixture.SHAPE_PREDICTOR)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Converting the image to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
            
                # Draw on our image, all the finded cordinate points (x,y) 
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Show the image
            cv2.imshow("Output", frame)

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