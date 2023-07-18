#!/usr/bin/python3

import cv2
import os

STATIC_THRESHOLD_MSECS = 3000

VIDEO_NAME = 'jarkom_9'
VIDEO_EXT = '.mp4'
VIDEO_PATH = './videos/' + VIDEO_NAME + VIDEO_EXT

dir_name = './slides/' + VIDEO_NAME

cap = cv2.VideoCapture(VIDEO_PATH)


def crop(frame):
    return frame


frame = None
gray = None
anchor_frame = None
anchor_gray = None
anchor_frame_num = None
anchor_time = None


def reset():
    global frame, gray, anchor_frame, anchor_gray, anchor_frame_num, anchor_time
    frame = None
    gray = None
    anchor_frame = None
    anchor_gray = None
    anchor_frame_num = None
    anchor_time = None

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def msec_to_human_readable(time, delimiter="-"):
    sec = time / 1000
    return '%02d%s%02d%s%03d' % (sec / 60, delimiter, sec % 60, delimiter, time % 1000)


def slide_detector(
    video_name=None,
    video_ext=None,
    target_dir='./slides/',
):
    global VIDEO_NAME, VIDEO_EXT, VIDEO_PATH, dir_name, cap

    VIDEO_NAME = video_name,
    VIDEO_NAME = VIDEO_NAME[0]
    VIDEO_EXT = video_ext,
    VIDEO_EXT = VIDEO_EXT[0]
    VIDEO_PATH = './videos/' + VIDEO_NAME + VIDEO_EXT
    dir_name = target_dir + VIDEO_NAME
    cap = cv2.VideoCapture(VIDEO_PATH)

    make_dir_if_not_exist(dir_name)

    reset()

    while (True):
        ret, frame = cap.read()
        if frame is None:
            break
        frame = crop(frame)

        if (cap.get(cv2.CAP_PROP_POS_FRAMES) % 2000) == 0:
            time_str = msec_to_human_readable(cap.get(cv2.CAP_PROP_POS_MSEC))
            print("Current time: %s" % time_str)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        def set_anchor():
            global anchor_frame, anchor_gray, anchor_frame_num, anchor_time
            anchor_frame = frame
            anchor_gray = gray
            anchor_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            anchor_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        if anchor_frame is None:
            set_anchor()

        deltaframe = cv2.absdiff(anchor_gray, gray)
        threshold = cv2.threshold(deltaframe, 10, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None)

        if threshold.any():
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            if time - anchor_time > STATIC_THRESHOLD_MSECS:
                time_str = msec_to_human_readable(anchor_time)
                print("Static frame at time %s for %f seconds"
                      % (time_str, (time-anchor_time) / 1000.0))
                cv2.imwrite('%s/%s.jpg' % (dir_name, time_str), anchor_frame)
            set_anchor()

    cap.release()
    cv2.destroyAllWindows()
