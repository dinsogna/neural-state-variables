import sys
import os
import cv2
import shutil

def save_loss(version_folder, summary_folder):
    loss_file = None
    for f in os.listdir(version_folder):
        if f.startswith('events'):
            loss_file = f'{version_folder}/{f}'
    if loss_file == None:
        print(f'Failed to save loss history: could not find loss file in {version_folder}')
        return
    shutil.copyfile(loss_file, f'{summary_folder}/loss')
    print(f'Successfully saved loss history to {summary_folder}/loss')

def save_video(predictions_folder, summary_folder):
    framerate = 10
    images = []
    video_idx = None
    for f in os.listdir(predictions_folder):
        if video_idx == None or f.startswith(video_idx):
            images.append(f)
            video_idx = f[:f.index('_') + 1]
    if video_idx == None:
        print(f'Failed to save video: no predictions found in {predictions_folder}')
        return
    images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))
    image_path = os.path.join(predictions_folder, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(f'{summary_folder}/output.mp4', fourcc, framerate, (width, height))
    for image in images:
        image_path = os.path.join(predictions_folder, image)
        frame = cv2.imread(image_path)
        out.write(frame)
    out.release()
    print(f'Successfully saved video {video_idx[:-1]} to {summary_folder}/output.mp4')

def save_variables(variables_folder, summary_folder):
    var_file = None
    for f in os.listdir(variables_folder):
        var_file = os.path.join(variables_folder, f)
        shutil.copyfile(var_file, f'{summary_folder}/{f}')
        print(f'Successfully saved {f} to {summary_folder}/{f}')
    if var_file == None:
        print(f'Failed to save prediction variables: no variables found in {variables_folder}')
    return

if __name__ == '__main__':
    assert len(sys.argv) > 1
    log_folder = sys.argv[1]
    if len(sys.argv) == 2:
        version = 0
    else:
        version = sys.argv[2]
    version_folder = f'{log_folder}/lightning_logs/version_{version}'
    predictions_folder = f'{log_folder}/predictions'
    variables_folder = f'{log_folder}/variables'
    summary_folder = f'{log_folder}/summary'
    if not os.path.exists(summary_folder):
        print(f'Creating summary folder at {summary_folder}')
        os.makedirs(summary_folder)
    # save_loss(version_folder, summary_folder)
    save_video(predictions_folder, summary_folder)
    save_variables(variables_folder, summary_folder)