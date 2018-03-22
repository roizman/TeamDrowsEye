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
    def __init__(self, detector, predictor):
        self.frame_counter = 0
        self.detector = detector
        self.predictor = predictor

    def push_frame(self, frame):

        if self.is_state_triggered(frame):
            self.frame_counter += 1
        else:
            self.frame_counter = 0

    def is_above_threshold(self):
        pass

    def is_state_triggered(self, gray_image):
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
    def __init__(self, detector, predictor):
        super(EyesShutDetector, self).__init__(detector, predictor)

    def is_state_triggered(self, frame):
        left_eye, right_eye = self.get_eyes_from_frame(frame, self.detector, self.predictor)

        if left_eye is not None:
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                return True

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
    def __init__(self, detector, predictor):
        super(JawDirectionDetector, self).__init__(detector, predictor)
        self.is_good = True

    def is_state_triggered(self, frame):
        jaw = self.get_jaw_from_frame(frame, self.detector, self.predictor)
        left_eye, right_eye = EyesShutDetector.get_eyes_from_frame(frame, self.detector, self.predictor)

        for point in range(1, len(jaw)):
            ptA = tuple(jaw[point - 1])
            ptB = tuple(jaw[point])
            cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

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
            self.is_good = False
        elif eyeRatio < leftThreshold:
            self.is_good = False
        else:
            self.is_good = True

    def is_above_threshold(self):
        return self.is_good

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
