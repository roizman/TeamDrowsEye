# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from logging.config import dictConfig

import os
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video.filevideostream import FileVideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import logging

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


# consts
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def main():
    # construct the argument parse and parse the arguments
    args = parse_cli_args()

    EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH = calculate_eye_aspect_ratio_thresholds()

    detector, predictor = init_dlib_detector_and_predictor(args["shape_predictor"])

    # start the video stream thread
    vs = None
    # if args[WEBCAM]:
    # log.debug("Starting video stream thread...")
    # vs = VideoStream(src=args["webcam"]).start()
    # time.sleep(1.0)
    # elif args[VIDEO_FILE]:
    log.debug("Grabbing video stream from file in {0}".format(args["video_file"]))
    if not os.path.exists(args["video_file"]):
        raise Exception("Video file doesn't exist")
    vs = cv2.VideoCapture(args["video_file"])
    # vs = FileVideoStream(args["video_file"], queueSize=2048)
    # vs.start()
    log.debug("is file opened? " + str(vs.isOpened()))
    # time.sleep(1)

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        log.debug("reading frame")
        ret, frame = vs.read()
        log.debug("frame read! :)" + str(ret))
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # get biggest rectangle (which should be the driver
        rect = max(rects, key= lambda rect: rect.area())

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

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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


def init_dlib_detector_and_predictor(shape_predictor):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    log.debug("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    return detector, predictor


def calculate_eye_aspect_ratio_thresholds():
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 10
    return EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH


def parse_cli_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor",  # required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=str, default="",
                    help="path alarm .WAV file")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("-w", WEBCAM, type=int, default=0,
                        help="index of webcam on system")
    source.add_argument("-f", VIDEO_FILE, type=str, default="",
                        help="Path to face video file")
    args = vars(ap.parse_args())
    # todo fix this, why isn't -p working?
    args["shape_predictor"] = r"sources\shape_predictor_68_face_landmarks.dat"
    args["video_file"] = r"sources\sample2.avi"
    return args


if __name__ == '__main__':
    main()
