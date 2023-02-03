#!/usr/local/bin/python3

import cv2
import argparse
import os
import sys

#parameters
framerate = 10.0

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = args['path']
ext = args['extension']
output = args['output']

images = []
if "_1" in dir_path:
    video_idx = "4_"
    output = "output_1.mp4"
if "_2" in dir_path:
    video_idx = "17_"
    output = "output_2.mp4"
if "_3" in dir_path:
    video_idx = "7_"
    output = "output_3.mp4"
for f in os.listdir(dir_path):
    if f.endswith(ext) and (video_idx == "_" or f.startswith(video_idx)):
        images.append(f)
        video_idx = f[:f.index('_') + 1]

# print(images)

# Sort the files found in the directory
# int_name = images[0].split(".")[0]
images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))
# print(images)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
# cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, framerate, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    # cv2.imshow('video',frame)
    # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
    #     break

# Release everything if job is finished
out.release()
# cv2.destroyAllWindows()

print("The output video is {}".format(output))

# print(f"Exported {output} to local machine")
