import sys
import os
import shutil
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python save_variables.py <Log Directory> <Save Directory>')
        print('   Ex: python save_variables.py scripts/logs_circular_motion_refine-64_1 logs_variables/constant_lambda_refine/lambda=0.01_1')
        exit(0)

    log = os.path.join(sys.argv[1], 'variables')
    save_log = sys.argv[2]
    if not os.path.isdir(save_log):
        os.makedirs(save_log)

    for f in os.listdir(log):
        path = os.path.join(log, f)
        data = np.load(path)
        save_path = os.path.join(save_log, f)
        np.save(save_path, data)
    
    log = os.path.join(sys.argv[1], 'lightning_logs/version_0')
    files = [f for f in os.listdir(log) if os.path.isfile(os.path.join(log, f)) and f.startswith('events')]
    assert(len(files) == 1)
    path = os.path.join(log, files[0])
    shutil.copy(path, os.path.join(save_log, 'loss'))
