"""
Convert a video to a sequence of images
"""
from __future__ import print_function, division

import argparse

import cv2
import numpy as np
from tqdm import tqdm

from constants import RIGHT_KEY, LEFT_KEY, ENTER_KEY, EXIT_KEYS, LEFT_KEY_MAC, RIGHT_KEY_MAC

parser = argparse.ArgumentParser(description='Split a video into a sequence of images')
parser.add_argument('-i', '--input_video', help='Input Video', default="", type=str, required=True)
parser.add_argument('-o', '--output_folder', help='Output folder', default="", type=str, required=True)
parser.add_argument('--display', action='store_true', default=False, help='Display the images')

args = parser.parse_args()

output_folder = args.output_folder

# OpenCV 3.x.x compatibility
if not hasattr(cv2, 'cv'):
    # 0-based index of the frame to be decoded/captured next.
    image_zero_index = cv2.CAP_PROP_POS_FRAMES
    frame_count = cv2.CAP_PROP_FRAME_COUNT
else:
    image_zero_index = cv2.cv.CV_CAP_PROP_POS_FRAMES
    frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT

video_file = args.input_video
# Read the video
cap = cv2.VideoCapture(video_file)

current_idx = cap.get(image_zero_index)
n_frames = int(cap.get(frame_count))
print("{} frames".format(n_frames))

# Progress Bar
pbar = tqdm(total=n_frames)

while True:
    # Read next frame
    while True:
        flag, img = cap.read()
        if flag:
            break
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(image_zero_index, current_idx - 1)
            cv2.waitKey(1000)
            continue

    original_img = img.copy()
    if not args.display:
        key = RIGHT_KEY  # Next frame
        if current_idx == n_frames - 1:
            break
    else:
        cv2.imshow("current", img)
        key = cv2.waitKey(0) & 0xff

    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        exit()
    elif key in [LEFT_KEY, LEFT_KEY_MAC, RIGHT_KEY, RIGHT_KEY_MAC, ENTER_KEY]:
        current_idx += 1 if key in [RIGHT_KEY, RIGHT_KEY_MAC, ENTER_KEY] else -1
        current_idx = np.clip(current_idx, 0, n_frames - 1)
        # Save image
        path = '{}/{}.jpg'.format(output_folder, int(current_idx))
        cv2.imwrite(path, original_img)
        # print("Saved {}".format(int(current_idx)))
    pbar.update(1)

    cap.set(image_zero_index, current_idx)

pbar.close()
