import matplotlib
matplotlib.use('Agg')

import sys, os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

np.random.seed(42)
tmax, dt = 3, 0.01
t = np.arange(0, tmax, dt)
EDRIFT = 0.0002
fps = 20
di = int(1/fps/dt)
    
L = 1
m = 1
g = 9.81
r = 0.18

def deriv(y, t, L, m):
    theta, omega = y
    theta_dot = omega
    omega_dot = -g/L * np.sin(theta)
    return theta_dot, omega_dot
    
def calc_E(y):
    theta, omega = y.T
    V = -m * L * g * np.cos(theta)
    T = 0.5 * m * (L * omega)**2
    return T + V

def make_plot(ax, x, y, i, idx):
    ax.plot([0, x[i]], [0, y[i]], lw=2, c='k')
    c0 = Circle((0, 0), r/4, fc='k', zorder=10)
    c1 = Circle((x[i], y[i]), r, fc='b', ec='b', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)

    ax.set_xlim(-L-r, L+r)
    ax.set_ylim(-L-r, L+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(f'single_pendulum_tmp/{idx}/{i//di}.png', bbox_inches=None, pad_inches=0.0)
    plt.cla()

def main():
    theta_min, theta_max = -np.pi, np.pi
    omega_min, omega_max = -2.0, 2.0
   
    theta_all, omega_all = [], []
    num_videos = int(input('Enter number of videos to generate: '))
    for video in range(num_videos):
        if not os.path.exists(f'single_pendulum_tmp/{video}'):
            os.mkdir(f'single_pendulum_tmp/{video}')
        fails = 0
        while True:
            init_theta = np.random.random() * (theta_max - theta_min) + theta_min
            init_omega = np.random.random() * (omega_max - omega_min) + omega_min
            y0 = np.array([init_theta, init_omega])
            y = odeint(deriv, y0, t, args=(L, m))
            E = calc_E(y0)
            if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
                fails += 1
                # print(f'Failed {fails} times')
            else:
                # print('Success')
                break
        theta, omega = y[:,0], y[:,1]
        length_scale = 0.8
        x = length_scale * L * np.sin(theta)
        y = -length_scale * L * np.cos(theta)
         
        fig = plt.figure(figsize=(1.28,1.28))
        ax = fig.add_subplot(111)
        
        for i in range(0, t.size, di):
            make_plot(ax, x, y, i, video)
        plt.close()
        theta_all.append(theta[::di])
        omega_all.append(omega[::di])

        if (video+1) % 100 == 0:
            print(f'Generated {video+1} videos!')
    theta_all = np.array(theta_all)
    omega_all = np.array(omega_all)
    np.savez('vars.npz', theta=theta_all, omega=omega_all)
    print('Saved variables!')

if __name__ == '__main__':
    main()