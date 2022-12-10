#!/usr/local/bin/python3

import cv2
import argparse
import os

#parameters
framerate = 10.0

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
ap.add_argument("-p", "--path", required=False, default='/home/dmi/neural-state-variables/scripts/logs_circular_motion_refine-64_1/prediction_long_term/model_rollout/5', help="path containing images")
args = vars(ap.parse_args())

# Arguments
# dir_path = "scripts/logs_circular_motion_encoder-decoder-64_1/prediction_long_term/model_rollout/4"
# dir_path = "circular_motion/circular_motion/4"
# dir_path = "scripts/logs_circular_motion_refine-64_1/prediction_long_term/model_rollout/4"
# dir_path = "scripts/logs_circular_motion_encoder-decoder_1/prediction_long_term/model_rollout/4"
dir_path = args['path']
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# print(images)

#Sort the files found in the directory
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