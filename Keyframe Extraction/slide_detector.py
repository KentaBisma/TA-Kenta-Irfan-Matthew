#!/usr/bin/python3

import cv2
import os
import numpy as np

# Constants
STATIC_THRESHOLD_MSECS = 3000
VIDEO_NAME = 'pengcit_1'
VIDEO_EXT = '.mp4'
VIDEO_PATH = './videos/' + VIDEO_NAME + VIDEO_EXT

dirname = './slides/' + VIDEO_NAME

cap = cv2.VideoCapture(VIDEO_PATH)

# Global variables
frame = None
gray = None
hue = None
anchor_frame = None
anchor_gray = None
anchor_frame_num = None
anchor_time = None
anchor_hue = None

last_saved_time = None

slide_frame_found = False

def msec_to_human_readable(time):
    sec = time / 1000
    return '%02d-%02d-%03d' % (sec / 60, sec % 60, time % 1000)


# Define cropping boundaries
# Cropping the frames accurately by the slide's actual
# boundaries helps eliminate unwanted pixels that may interfere
# with the statistic calculation for the difference between frames.
# Which ultimately decides whether a frame is
# a different slide from its previous.
# This is done by grabbing the bounding box of the biggest detected contour.
# Assuming that it represents the actual boundaries of the presented slide
# since it is commonly shaped as a rectangle.
def define_crop(img, disable=False):
    global x, y, w, h
    if not disable:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            maxContour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(maxContour)
            framed_img = img.copy()
            rect = cv2.rectangle(framed_img, (x, y), (x + w, y + h), (0,0,255), 3)
            cv2.imwrite(dirname + '/frame.jpg', rect)

            # If grabbed framing width is significant for cropping
            if w / img.shape[1] < 0.9:
                slide_frame_found = True
    else:
        x, y, w, h = 0, 0, 0, 0


# Checks whether the returned contour exists
def crop(frame):
    if w == 0 or h == 0:
        return frame
    return frame[y:y+h, x:x+w]


# Grabs the middle frame of the video
# Assuming it is viable as a cropping reference
def get_mid_frame_for_cropping_reference():
    newCap = cv2.VideoCapture(VIDEO_PATH)
    frame_number = newCap.get(cv2.CAP_PROP_FRAME_COUNT)//2
    newCap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, randomFrame = newCap.read()
    newCap.release()
    return randomFrame


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


make_dir_if_not_exist(dirname)

define_crop(get_mid_frame_for_cropping_reference(), disable=False)

while (True):
    # Grab current frame
    ret, uncropped_frame = cap.read()

    # End of Video
    if uncropped_frame is None:
        break

    frame = crop(uncropped_frame)

    # Preprocess the frame
    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % 2000) == 0:
        time_str = msec_to_human_readable(cap.get(cv2.CAP_PROP_POS_MSEC))
        print("Current time: %s" % time_str)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # For slide transitions detection
    hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]

    # Set the "anchor" frame for difference measurement
    def set_anchor():
        global anchor_frame, anchor_gray, anchor_frame_num, anchor_time, anchor_uncropped, anchor_hue
        anchor_frame = frame
        anchor_uncropped = uncropped_frame
        anchor_gray = gray
        anchor_hue = hue
        anchor_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        anchor_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    if anchor_frame is None:
        set_anchor()

    # cv2.imshow("Main", uncropped_frame)

    # cv2.imshow("Anchor", anchor_uncropped)

    # Measure the difference between anchored frame and current frame
    deltaframe = cv2.absdiff(anchor_gray, gray)

    # cv2.imshow("Delta", deltaframe)
    # cv2.waitKey(10)

    threshold = cv2.threshold(deltaframe, 10, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None)

    # cv2.imshow("Threshold", threshold)

    # If something caught above the defined treshold
    if threshold.any():
        time = cap.get(cv2.CAP_PROP_POS_MSEC)

        flag_sensitive = False
        if not slide_frame_found:
            # Region sensitive difference check
            size = np.size(threshold)
            crunch = np.sum(threshold)//255

            # Slide transition check
            anchor_hue = cv2.GaussianBlur(anchor_hue, (21, 21), 0)
            hue = cv2.GaussianBlur(hue, (21, 21), 0)

            hue_diff = cv2.threshold(cv2.absdiff(anchor_hue, hue), 128, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow("Anchor Hue", anchor_hue)
            # cv2.imshow("Hue", hue)
            # cv2.imshow("Hue Difference", flag)
            # cv2.waitKey(10)
            flag_sensitive = crunch/size >= 0.3 and hue_diff.any()

        flag_timer = time - anchor_time > STATIC_THRESHOLD_MSECS
        flag_last_set = last_saved_time is None
        flag_timer_2 = last_saved_time is not None and time - last_saved_time > STATIC_THRESHOLD_MSECS
        
        if flag_timer or (flag_sensitive and (flag_last_set or flag_timer_2)):
            time_str = msec_to_human_readable(anchor_time)
            print("Static frame at time %s for %f seconds"
                  % (time_str, (time-anchor_time) / 1000.0))
            cv2.imwrite(dirname + '/slide_at_' +
                        time_str + '.jpg', anchor_uncropped)
            last_saved_time = time
        set_anchor()

cap.release()
cv2.destroyAllWindows()
