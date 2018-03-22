# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

import argparse
import logging
import os
import time
# import the necessary packages
from logging.config import dictConfig
from threading import Thread

import cv2
import dlib
import imutils
import playsound
from imutils.video import VideoStream

from FacialStateDetector import EyesShutDetector, JawDirectionDetector, distance

SOURCES = "sources"

VIDEO_FILE = "--video_file"

WEBCAM = "--webcam"


def init_logger():
    logging_config = dict(
        version=1,
        formatters={
            'f': {'format':
                      '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': logging.DEBUG}
        },
        root={
            'handlers': ['h'],
            'level': logging.DEBUG,
        },
    )

    dictConfig(logging_config)

    return logging.getLogger()


log = init_logger()


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def main():
    # construct the argument parse and parse the arguments
    args = parse_cli_args()

    detector, predictor = init_dlib_detector_and_predictor(args["shape_predictor"])

    # start the video stream thread
    if "video_file" in args:
        log.debug("Grabbing video stream from file in {0}".format(args["video_file"]))
        if not os.path.exists(args["video_file"]):
            raise Exception(args["video_file"] + " doesn't exist!")
        vs = cv2.VideoCapture(args["video_file"])
        log.debug("is file opened? " + str(vs.isOpened()))
    else:
        log.debug("Starting video stream thread...")
        vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)

    # loop over frames from the video stream
    eyes_detector = EyesShutDetector()
    jaw_detector = JawDirectionDetector()

    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        log.debug("reading frame")
        if "video_file" in args:
            ret, frame = vs.read()
            log.debug("frame read! :)" + str(ret))
        else:
            frame = vs.read()

        frame = imutils.resize(frame, width=450)
        jaw = JawDirectionDetector.get_jaw_from_frame(frame, detector, predictor)
        left_eye, right_eye = EyesShutDetector.get_eyes_from_frame(frame, detector, predictor)

        eyes_detector.push_frame(left_eye, right_eye)
        jaw_detector.push_frame(jaw, left_eye, right_eye)
        if eyes_detector.is_above_threshold() or jaw_detector.is_above_threshold():
            raise_alarm(args, frame)
        else:
            print("AWAKE!!!")

        draw_eyes_on_frame(frame, left_eye, right_eye)
        draw_jaw_on_frame(frame, jaw, left_eye, right_eye)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    # vs.stop()
    vs.release()


def raise_alarm(arguments, frame):
    # check to see if an alarm file was supplied,
    # and if so, start a thread to have the alarm
    # sound played in the background
    if arguments["alarm"] != "":
        t = Thread(target=sound_alarm,
                   args=(arguments["alarm"],))
        t.deamon = True
        t.start()
    # draw an alarm on the frame
    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_jaw_on_frame(frame, jaw, left_eye, right_eye):
    for point in range(1, len(jaw)):
        ptA = tuple(jaw[point - 1])
        ptB = tuple(jaw[point])
        cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

    jawStart = jaw[0]
    jawEnd = jaw[len(jaw) - 1]
    leftEyeEdge = left_eye[len(left_eye) - 3]
    rightEyeEdge = right_eye[0]

    cv2.line(frame, tuple(jawStart), tuple(rightEyeEdge), (0, 0, 255), 2)
    cv2.line(frame, tuple(jawEnd), tuple(leftEyeEdge), (0, 0, 255), 2)

    rightEyeDistance = distance(jawStart, rightEyeEdge)
    leftEyeDistance = distance(jawEnd, leftEyeEdge)
    eyeRatio = rightEyeDistance / leftEyeDistance
    rightThreshold = 3
    leftThreshold = (1 / float(rightThreshold))

    if eyeRatio > rightThreshold:
        cv2.putText(frame, "Right", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif eyeRatio < leftThreshold:
        cv2.putText(frame, "Left", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Center", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_eyes_on_frame(frame, left_eye, right_eye):
    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    if left_eye is not None and right_eye is not None:
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


def init_dlib_detector_and_predictor(shape_predictor):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    log.debug("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    return detector, predictor


def parse_cli_args():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=str, default=os.path.join(SOURCES, "alarm.wav"),
                    help="path alarm .WAV file")

    input_subparsers = ap.add_subparsers(help="Types of inputs (video sources)")
    camera_parser = input_subparsers.add_parser("cam")
    file_parser = input_subparsers.add_parser("file")

    camera_parser.add_argument("-w", WEBCAM, type=int, default=0, required=True,
                               help="index of webcam on system")
    file_parser.add_argument("-f", VIDEO_FILE, type=str, default="", required=True,
                             help="Path to face video file")
    args = vars(ap.parse_args())
    args["shape_predictor"] = os.path.join(SOURCES, "shape_predictor_68_face_landmarks.dat")
    return args


if __name__ == '__main__':
    main()
