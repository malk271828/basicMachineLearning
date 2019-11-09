from imutils import face_utils
import cv2
from tqdm import tqdm

# Machine Learning Libraries
import dlib

class landmarksExtractor():
    """
    Reference
    ----------
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
    """
    def __init__(self,
                 shape_predictor:str,
                 fileName):
        """
        :param fileName: If this argument is not a string, video stream will be opened.
        """
        if isinstance(fileName, str):
            self.cap = cv2.VideoCapture(fileName)
        else:
            self.cap = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)

    def getLandmarks(self, verbose=0):
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

                    # Draw on our image, all the finded cordinate points (x,y)
                    if verbose > 0:
                        for (x, y) in landmarks:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if verbose > 0:
                    # Show the image
                    cv2.imshow("Output", frame)

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