import os
import subprocess
from subprocess import Popen
import shlex
import multiprocessing
import signal, psutil
import re
import numpy as np

    # FURTHER THINGS TO DO
    #   Have the name of the screen reflect the name of the thing 
    #       - maybe add seed so you can run multiple
    #   save the output of the script in .tex files so they are easily accesible
    #   git from Kevin the adjustments for the smoothing
    #   run low cost tests and build up from there
    #   block gpu access for a while
        #implemented this for latent
        # to test first run it on 1 gpu 2,2 to see if that can work then run on 2 gpus

    #Need Kevin to Commit 
    #- visual state vars
    #- values of lambda
    #- 0.05 and 1
    #- epochs - 10 and 10


def run(seed1, seed2, num_train_latent, num_train_state, gpus):
    gpus = np.sort(gpus)
    #gpu_proc = -np.array((len(gpus))) #which process are on which gpus, not sure I need this
    gpu_name = ["None"]*len(gpus) #which names are on which

    print("gpus: ", gpus)
    #print(gpu_proc)
    print("gpus_cur_names: ",gpu_name)

    latent_p = [0]*num_train_latent

    for i in range(num_train_latent):
        directory1 = "./" + seed1 + "-" + str(i)
        os.makedirs(directory1, exist_ok = True)

        full = True
        while full:
            for j in range(len(gpus)):
                if check_running(gpu_name[j]) == False:
                    command = '../scripts/dum.sh circular_motion ' + str(gpus[j]) + ' TRAIN_DUM-' + str(i)
                    gpu_name[j] = 'TRAIN_DUM-' + str(i) #update gpu name
                    #run latent training script
                    latent_p[i] = Popen(shlex.split(command), cwd=directory1)
                    full = False
                    print("latent: ", i)

        

    state_p = [0]*(num_train_state*num_train_latent + 1000) # fix this
    unfinshed_latent = list(np.arange(num_train_latent))
    while(len(unfinshed_latent) !=0):
        for i in unfinshed_latent:
            name = "TRAIN_DUM-" + str(i)
            if not check_running(name):
                unfinshed_latent.remove(i)

                #run the other ones
                for j in range(num_train_state):
                    print("latent: ", i, ", ", j)
                    directory2 = seed2 + "-" + str(j)

                    os.makedirs(directory1+"/" + directory2, exist_ok = True )
                    directory =  directory1 + '/' + directory2

                    gpu_num = wait_until_free(gpu_name)
                    print(gpu_name, gpu_num)
                    #command = "../../scripts/dum2.sh circular_motion " + str(gpus[0]) +  " TRAIN_DUM2-" + str(i)
                    command = '../../scripts/dum2.sh circular_motion ' + str(gpus[gpu_num]) + ' TRAIN_DUM2-' + str(i)
                    state_p[i * num_train_latent + j] =  Popen(shlex.split(command), cwd=directory)
                    gpu_name[gpu_num] = 'TRAIN_DUM2-' + str(i)

def wait_until_free(gpu_name): # return index when free

    free = False
    index = -1
    while not free:
        for i in range(len(gpu_name)):
            if check_running(gpu_name[i]) == False:
                index = i
                free = True

    return index


def check_running(name):
    cmd = "screen -ls | awk '/\." + name +  "\\t/ {print strtonum($1)}'"
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    all_run = re.findall(r'\d+', str(output))
    if len(all_run) == 1:
        return True
    else: # if it is 0 - then we are finished and if its more than 1 then its bad
        if len(all_run) >= 1:
            print("ERROR: multiple screens with same name")
        return False 

def child_processes(parent_pid, sig=signal.SIGTERM):
    try:
      parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
      return
    #children = parent.children(recursive=True)
    children = parent.get_children()
    return children
    #for process in children:
     # process.send_signal(sig)

#testing script
run("testing_c", "runa", 2, 2, [4, 5])