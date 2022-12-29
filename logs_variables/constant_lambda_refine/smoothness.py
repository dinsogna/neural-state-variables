import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_smoothness(refine_latent):
    num_vids = int(len(refine_latent) / 57)
    smoothness = 0.0
    for idx in range(num_vids):
        var1 = refine_latent[idx * 57: 57 * (idx + 1), 0]
        var2 = refine_latent[idx * 57: 57 * (idx + 1), 1]
        frames = len(var1) - 1
        for k in range(frames):
            smoothness += (var1[k+1] - var1[k])**2
            smoothness += (var2[k+1] - var2[k])**2
    return smoothness / num_vids

if __name__ == '__main__':
    N = 3
    lds = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    smoothness = np.zeros((len(lds), N))
    for idx, ld in enumerate(lds):
        for n in range(N):
            path = f'lambda={ld}_{n+1}'
            refine_latent = np.load(os.path.join(path, 'refine_latent.npy'))
            smoothness[idx][n] = calculate_smoothness(refine_latent)
    avg_smoothness = np.round(np.sum(smoothness, axis=1) / N, 2)
    std = np.round(np.std(smoothness, axis=1), 2)
    for k in range(len(lds)):
        print(f'lambda = {lds[k]} \t| smoothness = {avg_smoothness[k]} ± {std[k]}')
