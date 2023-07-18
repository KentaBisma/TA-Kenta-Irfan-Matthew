#!/usr/bin/python3

import random
import cv2
import os
import numpy as np
from asyncio import run

# Constants
VIDEO_NAME = 'jarkom_9'
VIDEO_EXT = '.mp4'
VIDEO_PATH = './videos/' + VIDEO_NAME + VIDEO_EXT

UI = False
SAVE_CROPPING_REFERENCE = False
SAVE_CROPPED = False

FRAMES_PER_LOG = 2000

WHITE_THRESHOLD = 240
CROPPING_STREL = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

GAUSSIAN_KERNEL = 21

TIME_THRESHOLD = 3000
GRAY_THRESHOLD = 10
HUE_THRESHOLD = 128
GLOBALITY_THRESHOLD = 0.0625

dir_name = './slides/' + VIDEO_NAME

cap = cv2.VideoCapture(VIDEO_PATH)

crop_reference = None

# Utility Functions


def msec_to_human_readable(time, delimiter="-"):
    sec = time / 1000
    return '%02d%s%02d%s%03d' % (sec / 60, delimiter, sec % 60, delimiter, time % 1000)


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_images(images_dict):

    # Create a list of images and labels
    images = []
    labels = []
    for key in images_dict:
        images.append(images_dict[key])
        labels.append(key)

    grid_size = (4, 2)

    width = images[0].shape[1] // 3
    height = images[0].shape[0] // 3

    cell_size = (width, height)

    # Create a default black image for padding empty slots in the grid
    black_image = np.full(
        (cell_size[0], cell_size[1], 3), dtype=np.uint8, fill_value=128)

    # Fill any empty slots in the images list with the black image
    num_images = len(images)
    num_cells = grid_size[0] * grid_size[1]
    if num_images < num_cells:
        images += [black_image] * (num_cells - num_images)

    # Resize the images to the same size
    images_resized = []
    for img in images:
        if (img.ndim == 2):
            img = np.stack((img,) * 3, axis=-1)
        img_resized = cv2.resize(img, cell_size)
        images_resized.append(img_resized)

    # Create the grid of images
    rows = []
    for i in range(0, num_cells, grid_size[0]):
        row = np.hstack(images_resized[i:i+grid_size[0]])
        rows.append(row)
    grid = np.vstack(rows)

    for i, label in enumerate(labels):
        x = i % grid_size[0]
        y = i // grid_size[0]
        cv2.putText(grid, label, (x*cell_size[0]+5, y*cell_size[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the grid with labels
    cv2.imshow("Images Grid", grid)
    cv2.waitKey(1)

# Define cropping area based on the largest contour's bounding box
# of the chosen frames.
def define_crop(frames, disable=False):
    global crop_reference
    if not disable:

        contours_frame_pair = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, CROPPING_STREL)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contours_frame_pair.append(
                    (frame, max(contours, key=cv2.contourArea)))

        def compare(el): return cv2.contourArea(el[1])
        max_pair = max(contours_frame_pair, key=compare)
        x, y, w, h = cv2.boundingRect(max_pair[1])
        crop_reference = {
            "x": x,
            "y": y,
            "w": w,
            "h": h
        }
        framed_img = max_pair[0].copy()
        rect = cv2.rectangle(framed_img, (x, y),
                             (x + w, y + h), (0, 0, 255), 3)
        if (SAVE_CROPPING_REFERENCE):
            cv2.imwrite(dir_name + '/frame.jpg', rect)

# Grabs the middle frame of the video
# Assuming it is viable as a cropping reference


def get_cropping_references():
    frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_nums = random.sample(range(0, frame_nums), 5)
    print('Chosen frames:%s' % random_frame_nums)
    ret = []
    for num in random_frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        _, random_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret.append(random_frame)
    return ret

# Crops the current frame


def crop(frame):
    if crop_reference is None:
        return frame
    y = crop_reference["y"]
    h = crop_reference["h"]
    x = crop_reference["x"]
    w = crop_reference["w"]
    return frame[y:y+h, x:x+w]

# Async-ly save a frame as an image


async def async_save_frame(name, frame):
    cv2.imwrite(name, frame)


def save_frame(name, frame):
    run(async_save_frame(name, frame))


"""
=====
Start
=====
"""

start = False
last_save = None

ui_images = dict()

make_dir_if_not_exist(dir_name)
if SAVE_CROPPED:
    make_dir_if_not_exist(dir_name + "_cropped")

define_crop(get_cropping_references())

while (True):
    # Capture current frame.
    _, current_frame = cap.read()  # TODO

    # End of Video.
    if current_frame is None:
        break

    cropped_frame = crop(current_frame)  # TODO

    # Logs current frame position per configured frame.
    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % FRAMES_PER_LOG) == 0:
        print("Current time: %s" % msec_to_human_readable(
            cap.get(cv2.CAP_PROP_POS_MSEC), delimiter=":"))

    # Extract needed information from the frame.
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # TODO
    ui_images["Gray"] = gray
    gray = cv2.GaussianBlur(
        gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)  # TODO
    ui_images["Gray Blurred"] = gray

    hue = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)[:, :, 0]  # TODO
    ui_images["Hue"] = hue
    hue = cv2.GaussianBlur(hue, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)  # TODO
    ui_images["Hue Blurred"] = hue

    # Inner function to set anchor information.
    def set_anchor():
        global anchor_frame, anchor_cropped_frame, anchor_gray, anchor_hue, anchor_time
        anchor_frame = current_frame
        anchor_cropped_frame = cropped_frame
        anchor_gray = gray
        anchor_hue = hue
        anchor_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        ui_images["Anchor"] = anchor_frame

    # Set the first anchor on the first frame.
    if not start:
        set_anchor()
        start = True

    """
    =========================
    Slide detection algorithm
    =========================
    """

    # Threshold the measured difference in gray to get binary values.
    _, gray_diff = cv2.threshold(cv2.absdiff(
        anchor_gray, gray), GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)  # TODO
    gray_diff = cv2.dilate(gray_diff, None)  # TODO

    # Check if something is different from gray perspective.
    flag = gray_diff.any()

    # If something was caught different enough from the anchor...
    if flag:
        # Check whether the difference happened after a defined interval.
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        flag_static = current_time - anchor_time > TIME_THRESHOLD

        if not flag_static:
            ui_images["Gray Diff"] = gray_diff

            # Check whether the difference is 'global' enough.
            _, binary_gray_diff = cv2.threshold(
                gray_diff, 0, 1, cv2.THRESH_BINARY)
            flag_globality = np.sum(binary_gray_diff) / \
                np.size(binary_gray_diff) >= GLOBALITY_THRESHOLD

            if flag_globality:
                # Threshold the measured difference in hue for slide transition check.
                _, hue_diff = cv2.threshold(cv2.absdiff(
                    anchor_hue, hue), HUE_THRESHOLD, 255, cv2.THRESH_BINARY)  # TODO
                hue_diff = cv2.dilate(hue_diff, None)  # TODO

                ui_images["Hue Diff"] = hue_diff

                # Check whether the difference does not indicate a slide transition.
                flag_is_not_transition = hue_diff.any()

            flag_save_timer = last_save is None or current_time - last_save > TIME_THRESHOLD

            flag_dynamic = flag_globality and flag_is_not_transition and flag_save_timer

        # If the difference is significant, save the frame and set the frame as the new anchor.
        if flag_static or flag_dynamic:
            print("Captured frame at time %s for %f seconds" % (msec_to_human_readable(
                anchor_time, delimiter=":"), (current_time - anchor_time) / 1000.0))
            save_frame('%s/%s.jpg' % (dir_name,
                       msec_to_human_readable(anchor_time, delimiter="-")), anchor_frame)
            if SAVE_CROPPED:
                save_frame('%s/%s.jpg' % (dir_name + "_cropped", msec_to_human_readable(
                    anchor_time, delimiter="-")), anchor_cropped_frame)

            last_save = current_time
            ui_images["Last Saved"] = anchor_frame

        set_anchor()

    if UI:
        plot_images(ui_images)


cap.release()
cv2.destroyAllWindows()
