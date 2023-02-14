import os
from PIL import Image

num_videos = 1200
num_frames = 60
for video in range(num_videos):
    if not os.path.exists(f'single_pendulum/{video}'):
        os.mkdir(f'single_pendulum/{video}')
    for frame in range(num_frames):
        image = Image.open(f'single_pendulum_tmp/{video}/{frame}.png').convert('RGB')
        image.save(f'single_pendulum/{video}/{frame}.png')
    if (video+1) % 100 == 0:
        print(f'Processed {video+1} videos!')