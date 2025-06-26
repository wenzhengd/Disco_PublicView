import numpy as np
#from qutip import *
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
import sys
from scipy import interpolate
from scipy.integrate import quad, dblquad,nquad
from scipy.linalg import expm
import matplotlib.pyplot as plt

import os
#os.chdir('/Users/wenzheng/Dropbox (Dartmouth College)/Code/check quantum noise')
#print("Current working directory: {0}".format(os.getcwd()))


#######################################
#      PARAMETERS
#######################################

GAMMA_FLIP = 0.01*10**6 # =0.01 Mhz //  flipping rate gamma
TIME_FINAL =20*10**-6 # =10 micro-second //  final time in noise_trajec 
NUM_STEPS = 1000 # num of steps in noise trajectory to TIME_FINAL
ENSEMBLE_SIZE = 1000 # size of noise realization

dt =10*10**-9  #= 1 nano-second  // small time step in whole numerics
T = 0.25*TIME_FINAL  # msmt time // T <= TIME_FINAL =10*10**-6 
num_steps =int(np.round(T/dt)) # number of time steps
time_list = [dt*i for i in range(0,num_steps)]


#######################################
# Iterate parameters !!!!!!!!!!!!!!!
#######################################

Omega_list = np.array([ 0, 0.4, 0.8 ])*10**6 
Tshift_list = np.array([0, 0.5, 2.0 ])* T  
g_list  = np.array([ 5,  10,   15,  20 ]) *GAMMA_FLIP


