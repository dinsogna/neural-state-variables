import os
import subprocess
from subprocess import Popen
import shlex
import multiprocessing
import signal, psutil
import re

def run(seed1, seed2, num_train_latent, num_train_state, gpus):

    #avaliable_gpus = gpus.copy() # at first all gpus are avaliable

    latent_p = [0]*num_train_latent
    #latent_staer

    for i in range(num_train_latent):
        directory1 = "./" + seed1 + "-" + str(i)
        os.makedirs(directory1, exist_ok = True)

        #run latent training script
        #os.system('cd ' + directory1 +  '; ../scripts/train_latent.sh circular_motion 4')
        command = '../scripts/train_latent_f.sh circular_motion 4'
        #command = '../scripts/dum.sh'
        latent_p[i] = Popen(shlex.split(command), cwd=directory1)
        print(i)

    state_p = [0]*num_train_state*num_train_latent
    latent_done = list()
    i = 0
    while (check_running(latent_p, num_train_latent)):
        i = 1
    while (check_running(latent_p, num_train_latent)):
        
        for i in range(num_train_latent):
            return_code = latent_p[i].poll()
            if (i not in latent_done) and (return_code != 0) and (return_code != None):
                print("return:", latent_p[i].poll())
                latent_done.append(i)

                for j in range(num_train_state):
                    print(i, ", ", j)
                    directory2 = seed2 + "-" + str(j)

                    os.makedirs(directory1+"/" + directory2, exist_ok = True )
                    directory =  directory1 + '/' + directory2
                    command = '../../scripts/train_refine_in.sh circular_motion 4'
                    state_p[i * num_train_latent + j] =  Popen(shlex.split(command), cwd=directory)

    print (check_running(latent_p, num_train_latent))

def check_running(p, i):
    cmd = "screen -ls | awk '/\.END-TO-END-TRAIN\\t/ {print strtonum($1)}'"
    print(cmd)
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    print(output)
    all_run = re.findall(r'\d+', str(output))
    print(all_run)
    if len(all_run) == 0:
        return False
    else:
        return True
    #pid = int(all_run[0])

    #for i in range(i):
        #if p[i].poll() == None or p[i].poll() == 0:
        #if p[i].poll() == None:
         #   return True
    #print("pid:", p[i].pid)
    #print("child-process:", child_processes(p[i].pid))
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

run("testing_c", "runa", 1, 2, 4)