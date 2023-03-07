import matplotlib
matplotlib.use('Agg')

import sys, os
import matplotlib
matplotlib.use('Agg')

import os
import shutil
import numpy as np
from scipy.integrate import solve_ivp

import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def mkdir(folder):
    if os.path.exists(folder):
        # shutil.rmtree(folder)
        pass
    os.makedirs(folder)

def engine(rng, num_frm, fps=60):
    m = 1.0
    l = 0.5
    g = 9.81
    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt

    f = lambda t, y: [y[1], -3*g/(2*l) * np.sin(y[0])]
    initial_state = [rng.uniform(-np.pi, np.pi), rng.uniform(-5, 5)]
    sol = solve_ivp(f, [t_eval[0], t_eval[-1]], initial_state, t_eval=t_eval, rtol=1e-6)
    
    states = sol.y.T
    return states

def render(ax, theta, image_filepath):
    L = 0.8
    r = 0.15
    border = 0.2
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    
    ax.plot([0, x], [0, y], lw=2, c='k')
    c0 = Circle((0, 0), r/4, fc='k', zorder=10)
    c1 = Circle((x, y), r, fc='b', ec='b', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)

    ax.set_xlim(-L-r-border, L+r+border)
    ax.set_ylim(-L-r-border, L+r+border)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches=None, pad_inches=0.0)
    im = Image.open(img_buf).convert('RGB')
    im.save(image_filepath)
    plt.cla()

def make_data(data_filepath, num_seq, num_frm, seed=0):
    mkdir(data_filepath)
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 2))
    
    fig = plt.figure(figsize=(1.28,1.28))
    ax = fig.add_subplot(111)
    
    for n in range(num_seq):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            render(ax, states[n, k, 0], os.path.join(seq_filepath, str(k)+'.png'))
        if (n+1) % 10 == 0:
            print(f'Generated {n+1} videos!')
    
    np.save('./states.npy', states)

if __name__ == '__main__':
    data_filepath = './single_pendulum'
    make_data(data_filepath, num_seq=1200, num_frm=60)