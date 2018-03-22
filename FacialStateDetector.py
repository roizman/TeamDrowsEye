import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jaw_start, jaw_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]


class FacialStateDetector(object):
    def __init__(self):
        self.frame_counter = 0

    def push_frame(self, *args):

        if self.is_state_triggered(*args):
            self.frame_counter += 1
        else:
            self.frame_counter = 0

    def is_above_threshold(self):
        pass

    def is_state_triggered(self, *args):
        pass


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


class EyesShutDetector(FacialStateDetector):

    def is_state_triggered(self, *args):
        left_eye = args[0]
        right_eye = args[1]
        if left_eye is not None:
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                return True
        return False

    def is_above_threshold(self):
        return self.frame_counter > EYE_AR_CONSEC_FRAMES

    @staticmethod
    def get_eyes_from_frame(frame, detector, predictor):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # get biggest rectangle (which should be the driver
        rect = max(rects, key=lambda rect1: rect1.area()) if rects else None

        left_eye, right_eye = None, None
        # loop over the face detections
        if rect:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

        return left_eye, right_eye


class JawDirectionDetector(FacialStateDetector):
    def __init__(self):
        super(JawDirectionDetector, self).__init__()
        self._is_good = True

    def is_state_triggered(self, *args):
        jaw = args[0]
        left_eye = args[1]
        right_eye = args[2]

        jawStart = jaw[0]
        jawEnd = jaw[len(jaw) - 1]

        leftEyeEdge = left_eye[len(left_eye) - 3]
        rightEyeEdge = right_eye[0]

        rightEyeDistance = distance(jawStart, rightEyeEdge)
        leftEyeDistance = distance(jawEnd, leftEyeEdge)
        eyeRatio = rightEyeDistance / leftEyeDistance
        rightThreshold = 3
        leftThreshold = (1 / float(rightThreshold))

        if eyeRatio > rightThreshold:
            self._is_good = False
        elif eyeRatio < leftThreshold:
            self._is_good = False
        else:
            self._is_good = True

    def is_above_threshold(self):
        return not self._is_good

    @staticmethod
    def get_jaw_from_frame(frame, detector, predictor):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # get biggest rectangle (which should be the driver
        rect = max(rects, key=lambda rect1: rect1.area()) if rects else None

        jaw = None
        # loop over the face detections
        if rect:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            jaw = shape[jaw_start:jaw_end]

        return jaw


def distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) * 2) + ((point1[1] - point2[1]) * 2))
