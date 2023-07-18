from slide_detector_kenta_eval import slide_detector_kenta
from slide_detector_eval import slide_detector

VIDEO_NAME = ''
VIDEO_EXT = '.mp4'

THRESHOLDS = []

# VIDEO = cv2.VideoCapture('./videos/' + VIDEO_NAME + VIDEO_EXT)

TP = 0
GTP = 0
ETP = 0

# EXTRACTED_VIDEOS = ['basdat_4', 'basdat_10',
#                     'jarkom_6', 'jarkom_8', 'jarkom_9']

# for video in EXTRACTED_VIDEOS:
#     slide_detector_kenta(
#         video_name=video,
#         video_ext='.mp4',
#         target_dir='./eval/extracted/slide-detector-kenta/',
#         gaussian_kernel=21,
#         white_threshold=240,
#         time_threshold=3000,
#         gray_threshold=10,
#         hue_threshold=128,
#         globality_threshold=.0625,
#     )
#     slide_detector(
#         video_name=video,
#         video_ext='.mp4',
#         target_dir='./eval/extracted/slide-detector/',
#     )

slide_detector_kenta(
    video_name='pengcit_1',
    video_ext='.mp4',
    target_dir='./eval/extracted/slide-detector-kenta/',
    gaussian_kernel=21,
    white_threshold=240,
    time_threshold=3000,
    gray_threshold=10,
    hue_threshold=128,
    globality_threshold=.0625,
)