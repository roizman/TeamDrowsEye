import argparse
import time

import cv2
import dlib
import imutils
import numpy as np
import playsound
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist



def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


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


def distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",  # required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

args["shape_predictor"] = "sources/shape_predictor_68_face_landmarks.dat"
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = True

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jaw_start, jaw_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# frameHeight = 1200
# frameWidth = 680

print("Grabbing video stream from file in {0}".format("sources/sample3.avi"))
file_vs = cv2.VideoCapture("sources/sample4.avi")
print("is file opened? " + str(file_vs.isOpened()))

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)

    ret, video_frame = file_vs.read()

    frame = vs.read()
    frame = imutils.resize(frame, height=300)

    frame = cv2.flip(frame, 1)

    #detect_logos(video_frame)
    video_frame = imutils.resize(video_frame, height=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    print frame.shape[1]
    print frame.shape[0]

    print video_frame.shape[1]
    print video_frame.shape[0]
    """

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # jaw = shape[jaw_start:jaw_end]

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        nose = shape[noseStart:noseEnd]

        jaw = shape[jaw_start:jaw_end]
        jawStart = jaw[0]
        jawEnd = jaw[len(jaw) - 1]

        leftEyeEdge = leftEye[len(leftEye) - 3]
        rightEyeEdge = rightEye[0]

        rightEyeDistance = distance(jawStart, rightEyeEdge)
        leftEyeDistance = distance(jawEnd, leftEyeEdge)
        eyeRatio = rightEyeDistance / leftEyeDistance
        rightThreshold = 5
        leftThreshold = (1 / float(rightThreshold))
        bottomRight = (int(0.9 * frameWidth), int(0.7 * frameHeight))
        bottomMiddle = (int(0.45 * frameWidth), int(0.7 * frameHeight))
        bottomLeft = (int(0.075 * frameWidth), int(0.7 * frameHeight))
        yawnThreshold = int(0.06 * frameHeight)

        mouth = shape[mouthStart:mouthEnd]
        mouthTop = mouth[3]
        mouthBottom = mouth[9]
        cv2.line(frame, tuple(mouthTop), tuple(mouthBottom), (0, 0, 255), 2)
        # print distance(mouthTop, mouthBottom)

        for point in range(1, len(mouth)):
            ptA = tuple(mouth[point - 1])
            ptB = tuple(mouth[point])
            cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.line(frame, tuple(jawStart), tuple(rightEyeEdge), (0, 0, 255), 2)
        cv2.line(frame, tuple(jawEnd), tuple(leftEyeEdge), (0, 0, 255), 2)

        for point in range(1, len(jaw)):
            ptA = tuple(jaw[point - 1])
            ptB = tuple(jaw[point])
            cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

        chinPoint = jaw[8]
        noseTip = nose[6]
        # leftEyeEdge
        # rightEyeEdge
        leftMouthCorner = mouth[0]
        rightMouthCorner = mouth[6]

        if True:
            frame_shape = frame.shape
            image_points = np.array([
                noseTip,
                chinPoint,
                leftEyeEdge,
                rightEyeEdge,
                leftMouthCorner,
                rightMouthCorner
            ], dtype="double")

            # 3D model points.
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -(330.0 / 675) * frameHeight, -65.0),  # Chin
                (-(225.0 / 1200) * frameWidth, (170.0 / 675) * frameHeight, -135.0),  # Left eye left corner
                ((225.0 / 1200) * frameWidth, (170.0 / 675) * frameHeight, -135.0),  # Right eye right corner
                (-(150.0 / 1200) * frameWidth, -(150.0 / 675) * frameHeight, -125.0),  # Left Mouth corner
                ((150.0 / 1200) * frameWidth, -(150.0 / 675) * frameHeight, -125.0)  # Right mouth corner
            ])

            focal_length = frame_shape[1]
            center = (frame_shape[1] / 2, frame_shape[0] / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")
            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            if eyeRatio > rightThreshold or eyeRatio < leftThreshold:
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            x = p2[0]
            y = p2[1]

            # print height video_frame.shape[0], width video_frame.shape[1]

            y_ratio = float(y) / float(frame.shape[0])
            x_ratio = float(x) / float(frame.shape[1])
            new_point = (int(video_frame.shape[1] * x_ratio), int(video_frame.shape[0] * y_ratio))
            cv2.circle(video_frame, new_point, 50, (0, 0, 255), 10)

            # print p1
    both = np.hstack((frame, video_frame))
    cv2.imshow('imgc', both)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
