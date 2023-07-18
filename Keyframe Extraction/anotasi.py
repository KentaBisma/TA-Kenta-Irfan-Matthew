from asyncio import run
import os
import cv2


def jaccard_index(annotations_list):
    timestamps_list = []

    for annotations in annotations_list:
        timestamps = set()
        for timestamp in annotations:
            for x in time_range(timestamp):
                timestamps.add(x)
        timestamps_list.append(timestamps)

    intersection = set.intersection(*timestamps_list)
    union = set.union(*timestamps_list)

    return len(intersection) / len(union)


def accumulate_lines(file_paths):
    line_dict = dict()

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = [*filter(lambda c: c != "", [line.replace("\n", "").replace(" ", "")
                                               for line in file.readlines()])]
            line_dict[str(file_path).split('/')[2][:-4]] = lines

    return line_dict


def time_range(timestamp):
    m, s = map(int, timestamp.split('-'))
    seconds = m * 60 + s

    return [f"{((seconds+x)//60):02d}-{((seconds+x)%60):02d}" for x in range(-1, 2)]


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


async def async_save_frame(name, frame):
    cv2.imwrite(name, frame)


def save_frame(name, frame):
    run(async_save_frame(name, frame))


sets = [
    "basdat_4", "basdat_10", "jarkom_6", "jarkom_8", "jarkom_9"
]

for set_name in sets:
    file_paths = [
        f'eval/annotations/{set_name}_{x}.txt' for x in ["Rafi", "kenta", "irfan"]]

    annotations = accumulate_lines(file_paths)

    print("Similarity for set:", set_name)
    print("Jaccard Index:", jaccard_index([*annotations.values()]))

    # for k, v in annotations.items():
    #     VIDEO_EXT = '.mp4'
    #     VIDEO = cv2.VideoCapture('./videos/' + set_name + VIDEO_EXT)
    #     PATH = f'./eval/frames/{k}/'
    #     make_dir_if_not_exist(PATH)
    #     for timestamp in v:
    #         m = int(timestamp.split("-")[0])
    #         s = int(timestamp.split("-")[1])
    #         msec = (s * 1000) + (m * 60 * 1000)

    #         # if msec != 0 : msec -= 

    #         VIDEO.set(cv2.CAP_PROP_POS_MSEC, msec)
    #         _, frame = VIDEO.read()

    #         save_frame(PATH + timestamp + '.jpg', frame)

    #     VIDEO.release()
