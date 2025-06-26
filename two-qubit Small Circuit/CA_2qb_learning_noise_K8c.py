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

from CA_params_setting import *

from C_2qb_qns_K8_L2_classical import *
#from C_2qb_qns_K4_L4_classical import *
#from C_2qb_qns_K2_L4_quantum import *
#from C_2qb_qns_K2_L4_classical import *

n_cores = 50


#######################################
# setups
#######################################

cos = np.cos
sin = np.sin
sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1= np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])

sigma_00 = np.kron(sigma_0,sigma_0) # two-qubit generators
sigma_01 = np.kron(sigma_0,sigma_1)
sigma_02 = np.kron(sigma_0,sigma_2)
sigma_03 = np.kron(sigma_0,sigma_3)
sigma_10 = np.kron(sigma_1,sigma_0)
sigma_11 = np.kron(sigma_1,sigma_1)
sigma_12 = np.kron(sigma_1,sigma_2)
sigma_13 = np.kron(sigma_1,sigma_3)
sigma_20 = np.kron(sigma_2,sigma_0)
sigma_21 = np.kron(sigma_2,sigma_1)
sigma_22 = np.kron(sigma_2,sigma_2)
sigma_23 = np.kron(sigma_2,sigma_3)
sigma_30 = np.kron(sigma_3,sigma_0)
sigma_31 = np.kron(sigma_3,sigma_1)
sigma_32 = np.kron(sigma_3,sigma_2)
sigma_33 = np.kron(sigma_3,sigma_3)
pauli_2_generator = [sigma_00,sigma_01,sigma_02,sigma_03,  sigma_10,sigma_11,sigma_12,sigma_13,  sigma_20,sigma_21,sigma_22,sigma_23,  sigma_30,sigma_31,sigma_32,sigma_33]

L = 2 # number of windows
dim_sys = 4 # dimensional of qubit-system


########################################
# Some essential functions
########################################

def RTN_generator():
	"""
	1. produce a zero mean noise // also stationary 
	2. Ff we know that state is s at time t: z_s(t), then at t+dt, the flip to s' has probablity
	P_flip(t, t+dt) = e^(-gamma dt)
    
    The noise is modulated  cos(Omega t +phi)
     
	"""
	TIME_NUMBER = int(np.round(TIME_FINAL/dt))
	trajectory_table = np.zeros((ENSEMBLE_SIZE,TIME_NUMBER))
	for i in range(ENSEMBLE_SIZE):
		trajectory_table[i][0] = 1 if (np.random.uniform(0,1)>0.5) else -1 #  \pm 1 full-random zero-mean
		j=1
		while j<TIME_NUMBER:
			trajectory_table[i][j] = 1 * trajectory_table[i][j-1] if ( GAMMA_FLIP*dt  < np.random.uniform(0, 1)) \
									else -1* trajectory_table[i][j-1]
			j+=1
    # now add cos modulation 
	for i in range(ENSEMBLE_SIZE):
		phi =  np.random.uniform(0, 1)*2*np.pi
		for j in range(TIME_NUMBER):
			trajectory_table[i][j] = trajectory_table[i][j] * np.cos(Omega * j * dt + phi)
	return trajectory_table

def const_noise_generator():
	"""
	1. produce a constant noise, == +1 
     
	"""
	ENSEMBLE_SIZE =2
	TIME_NUMBER = int(np.round(TIME_FINAL/dt))
	trajectory_table = np.zeros((ENSEMBLE_SIZE,TIME_NUMBER))
	for i in range(ENSEMBLE_SIZE):
		for j in range(TIME_NUMBER):
			trajectory_table[i][j] = 1 # 
	return trajectory_table

def const_0mean_noise_generator():
	"""
	1. produce a zero mean noise, == +1  & -1
     
	"""
	ENSEMBLE_SIZE =2
	TIME_NUMBER = int(np.round(TIME_FINAL/dt))
	trajectory_table = np.zeros((ENSEMBLE_SIZE,TIME_NUMBER))
	for i in range(ENSEMBLE_SIZE):
		for j in range(TIME_NUMBER):
			trajectory_table[i][j] = (-1)**i # 
	return trajectory_table
    

#def U_sb(i,j,tt): # Calculate AB_S joint system propagator based on y_q[n] and noise_trajectory
#	"""
#	i is C_ctrl index, j is realization index;
#	tt is time , 0<=tt<= T
#	"""
#	# in the following coeff(p,q,s) is the interp of y^[q]_p * B_u // p in [1,15], q in [1,2], s in [1,3]
#	dim_p, dim_q, dim_s  = 15,2,3 
#	coeff = np.zeros((dim_p, dim_q, dim_s))
#	for p in range(dim_p): # pauli_2 
#		for q in range(dim_q): # q=A,B
#			y_qns = y_A_qns if q==0 else y_B_qns
#			for s in range(dim_s): # x,y,z 
#				g_ = g_A_x if (q==0 and s ==0) else g_B_x if (q==1 and s ==0) else g_A_y  if(q==0 and s ==1) \
#					else g_B_y if(q==1 and s ==1) else g_A_z if(q==0 and s ==2)	else g_B_z
#				b_ = b_x[j, :] if (s==0) else b_y if (s==1) else b_z  # b_x is 2d-array, and we pick j-th row 
#				coeff[p,q,s] = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][p] \
#																* g_ * b_[k] for k in range(num_steps)])
#	steps = int(tt/T*(num_steps))   # if tt=T, steps = num_steps
#	matrix = np.kron(sigma_00,  sigma_0) # inital state of S-B total system
#	for k in range(num_steps):
#		matrix = expm(-1j*(tt/steps)*\
#			np.sum([np.kron( coeff[p,q,s](k/steps*tt)* pauli_2_generator[p] ,  (sigma_1 if s==0 else sigma_2 if s==1 else sigma_3)) \
#				for p in range(dim_p) for q in range(dim_q) for s in range(dim_s) ]) )  @ matrix
#	return matrix		


def U_sb(i,j, tt): #
	"""
	i is C_ctrl index, j is realization index;
	tt is time , 0<=tt<= T
	"""
	# in the following, coeff_u_v is interp of  {y_u(t) * b_v(t)} 
	coeff_1_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_1_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_1_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_1_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_1_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_1_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_2_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_2_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_2_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_2_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_2_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_2_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_3_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_3_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_3_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_3_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_3_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_3_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_4_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_4_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_4_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_4_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_4_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_4_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][3] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_5_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_5_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_5_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_5_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_5_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_5_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][4] * g_B_z*b_z[j,k] for k in range(num_steps)])		
	coeff_6_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_6_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_6_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_6_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_6_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_6_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][5] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_7_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_7_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_7_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_7_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_7_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_7_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][6] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_8_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_8_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_8_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_8_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_8_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_8_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][7] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_9_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_9_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_9_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_9_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_9_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_9_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][8] * g_B_z*b_z[j,k] for k in range(num_steps)])	
	coeff_10_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_10_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_10_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_10_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_10_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_10_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][9] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_11_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_11_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_11_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_11_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_11_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_11_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][10] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_12_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_12_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_12_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_12_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_12_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_12_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][11] * g_B_z*b_z[j,k] for k in range(num_steps)])		
	coeff_13_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_13_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_13_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_13_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_13_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_13_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][12] * g_B_z*b_z[j,k] for k in range(num_steps)])
	coeff_14_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_14_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_14_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_14_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_14_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_14_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][13] * g_B_z*b_z[j,k] for k in range(num_steps)])	
	coeff_15_A_x = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_A_x*b_x[j,k] for k in range(num_steps)])
	coeff_15_B_x = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_B_x*b_x[j,k] for k in range(num_steps)])
	coeff_15_A_y = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_A_y*b_y[j,k] for k in range(num_steps)])
	coeff_15_B_y = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_B_y*b_y[j,k] for k in range(num_steps)])
	coeff_15_A_z = interpolate.interp1d(time_list,[y_A_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_A_z*b_z[j,k] for k in range(num_steps)])
	coeff_15_B_z = interpolate.interp1d(time_list,[y_B_qns[i,int(np.floor(k/(num_steps/L)))][14] * g_B_z*b_z[j,k] for k in range(num_steps)])	
#	coeff_x_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_x[j,k] for k in range(num_steps)])  
#	coeff_x_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_y[j,k] for k in range(num_steps)])
#	coeff_x_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_z[j,k] for k in range(num_steps)])
#	coeff_y_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_x[j,k] for k in range(num_steps)])
#	coeff_y_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_y[j,k] for k in range(num_steps)])
#	coeff_y_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_z[j,k] for k in range(num_steps)])
#	coeff_z_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_x[j,k] for k in range(num_steps)])
#	coeff_z_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_y[j,k] for k in range(num_steps)])    
#	coeff_z_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_z[j,k] for k in range(num_steps)])
	steps = int(tt/T*(num_steps))   # if tt=T, steps = num_steps
	matrix = np.kron(np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),  np.matrix([[1,0],[0,1]])) # inital state of S-B total system
	for k in range(num_steps):
		matrix = expm(-1j*(tt/steps)*(np.kron(coeff_1_A_x(k/steps*tt)*pauli_2_generator[1]+coeff_1_B_x(k/steps*tt)*pauli_2_generator[1]+
									   coeff_2_A_x(k/steps*tt)*pauli_2_generator[2]+coeff_2_B_x(k/steps*tt)*pauli_2_generator[2]+
									   coeff_3_A_x(k/steps*tt)*pauli_2_generator[3]+coeff_3_B_x(k/steps*tt)*pauli_2_generator[3]+
									   coeff_4_A_x(k/steps*tt)*pauli_2_generator[4]+coeff_4_B_x(k/steps*tt)*pauli_2_generator[4]+
									   coeff_5_A_x(k/steps*tt)*pauli_2_generator[5]+coeff_5_B_x(k/steps*tt)*pauli_2_generator[5]+
									   coeff_6_A_x(k/steps*tt)*pauli_2_generator[6]+coeff_6_B_x(k/steps*tt)*pauli_2_generator[6]+
									   coeff_7_A_x(k/steps*tt)*pauli_2_generator[7]+coeff_7_B_x(k/steps*tt)*pauli_2_generator[7]+
									   coeff_8_A_x(k/steps*tt)*pauli_2_generator[8]+coeff_8_B_x(k/steps*tt)*pauli_2_generator[8]+
									   coeff_9_A_x(k/steps*tt)*pauli_2_generator[9]+coeff_9_B_x(k/steps*tt)*pauli_2_generator[9]+
									   coeff_10_A_x(k/steps*tt)*pauli_2_generator[10]+coeff_10_B_x(k/steps*tt)*pauli_2_generator[10]+
									   coeff_11_A_x(k/steps*tt)*pauli_2_generator[11]+coeff_11_B_x(k/steps*tt)*pauli_2_generator[11]+
									   coeff_12_A_x(k/steps*tt)*pauli_2_generator[12]+coeff_12_B_x(k/steps*tt)*pauli_2_generator[12]+
									   coeff_13_A_x(k/steps*tt)*pauli_2_generator[13]+coeff_13_B_x(k/steps*tt)*pauli_2_generator[13]+
									   coeff_14_A_x(k/steps*tt)*pauli_2_generator[14]+coeff_14_B_x(k/steps*tt)*pauli_2_generator[14]+
									   coeff_15_A_x(k/steps*tt)*pauli_2_generator[15]+coeff_15_B_x(k/steps*tt)*pauli_2_generator[15],sigma_1)+
									   np.kron(coeff_1_A_y(k/steps*tt)*pauli_2_generator[1]+coeff_1_B_y(k/steps*tt)*pauli_2_generator[1]+
									   coeff_2_A_y(k/steps*tt)*pauli_2_generator[2]+coeff_2_B_y(k/steps*tt)*pauli_2_generator[2]+
									   coeff_3_A_y(k/steps*tt)*pauli_2_generator[3]+coeff_3_B_y(k/steps*tt)*pauli_2_generator[3]+
									   coeff_4_A_y(k/steps*tt)*pauli_2_generator[4]+coeff_4_B_y(k/steps*tt)*pauli_2_generator[4]+
									   coeff_5_A_y(k/steps*tt)*pauli_2_generator[5]+coeff_5_B_y(k/steps*tt)*pauli_2_generator[5]+
									   coeff_6_A_y(k/steps*tt)*pauli_2_generator[6]+coeff_6_B_y(k/steps*tt)*pauli_2_generator[6]+
									   coeff_7_A_y(k/steps*tt)*pauli_2_generator[7]+coeff_7_B_y(k/steps*tt)*pauli_2_generator[7]+
									   coeff_8_A_y(k/steps*tt)*pauli_2_generator[8]+coeff_8_B_y(k/steps*tt)*pauli_2_generator[8]+
									   coeff_9_A_y(k/steps*tt)*pauli_2_generator[9]+coeff_9_B_y(k/steps*tt)*pauli_2_generator[9]+
									   coeff_10_A_y(k/steps*tt)*pauli_2_generator[10]+coeff_10_B_y(k/steps*tt)*pauli_2_generator[10]+
									   coeff_11_A_y(k/steps*tt)*pauli_2_generator[11]+coeff_11_B_y(k/steps*tt)*pauli_2_generator[11]+
									   coeff_12_A_y(k/steps*tt)*pauli_2_generator[12]+coeff_12_B_y(k/steps*tt)*pauli_2_generator[12]+
									   coeff_13_A_y(k/steps*tt)*pauli_2_generator[13]+coeff_13_B_y(k/steps*tt)*pauli_2_generator[13]+
									   coeff_14_A_y(k/steps*tt)*pauli_2_generator[14]+coeff_14_B_y(k/steps*tt)*pauli_2_generator[14]+
									   coeff_15_A_y(k/steps*tt)*pauli_2_generator[15]+coeff_15_B_y(k/steps*tt)*pauli_2_generator[15],sigma_2)+
									   np.kron(coeff_1_A_z(k/steps*tt)*pauli_2_generator[1]+coeff_1_B_z(k/steps*tt)*pauli_2_generator[1]+
									   coeff_2_A_z(k/steps*tt)*pauli_2_generator[2]+coeff_2_B_z(k/steps*tt)*pauli_2_generator[2]+
									   coeff_3_A_z(k/steps*tt)*pauli_2_generator[3]+coeff_3_B_z(k/steps*tt)*pauli_2_generator[3]+
									   coeff_4_A_z(k/steps*tt)*pauli_2_generator[4]+coeff_4_B_z(k/steps*tt)*pauli_2_generator[4]+
									   coeff_5_A_z(k/steps*tt)*pauli_2_generator[5]+coeff_5_B_z(k/steps*tt)*pauli_2_generator[5]+
									   coeff_6_A_z(k/steps*tt)*pauli_2_generator[6]+coeff_6_B_z(k/steps*tt)*pauli_2_generator[6]+
									   coeff_7_A_z(k/steps*tt)*pauli_2_generator[7]+coeff_7_B_z(k/steps*tt)*pauli_2_generator[7]+
									   coeff_8_A_z(k/steps*tt)*pauli_2_generator[8]+coeff_8_B_z(k/steps*tt)*pauli_2_generator[8]+
									   coeff_9_A_z(k/steps*tt)*pauli_2_generator[9]+coeff_9_B_z(k/steps*tt)*pauli_2_generator[9]+
									   coeff_10_A_z(k/steps*tt)*pauli_2_generator[10]+coeff_10_B_z(k/steps*tt)*pauli_2_generator[10]+
									   coeff_11_A_z(k/steps*tt)*pauli_2_generator[11]+coeff_11_B_z(k/steps*tt)*pauli_2_generator[11]+
									   coeff_12_A_z(k/steps*tt)*pauli_2_generator[12]+coeff_12_B_z(k/steps*tt)*pauli_2_generator[12]+
									   coeff_13_A_z(k/steps*tt)*pauli_2_generator[13]+coeff_13_B_z(k/steps*tt)*pauli_2_generator[13]+
									   coeff_14_A_z(k/steps*tt)*pauli_2_generator[14]+coeff_14_B_z(k/steps*tt)*pauli_2_generator[14]+
									   coeff_15_A_z(k/steps*tt)*pauli_2_generator[15]+coeff_15_B_z(k/steps*tt)*pauli_2_generator[15],sigma_3) 	))@ matrix
	return matrix

def U_sb_dag(i,j,tt):
	return U_sb(i,j,tt).getH()

def obs_avg(OO,i,rho_B = 1/2*(sigma_0+sigma_3)):
	"""
	<OO(T)>|_{rho_s,rho_b}
	where i tells which "ctrl" and which  qubit init_state
	rho_B is the intial_state of bath // for classical noise, can I let it be |+1> ? 
	T will be fixed at final time
	"""
	b_dim=4
	f_map = lambda s: pauli_2_generator[s-1] # map the inital state / s =2,..16
	exp_f =lambda j: np.trace(U_sb(i,j,T) @ np.kron(f_map(init_list[i]),rho_B) \
									@ U_sb_dag(i,j,T) @ np.kron(OO, sigma_0) )  # should be trace, not dime-trace Tr[__]/dim_sys
	obs_each = Parallel(n_jobs=n_cores, verbose=0)(delayed(exp_f)(j) for j in range (ENSEMBLE_SIZE) )
    
	return np.average(obs_each)-int(init_list[i]==16)* dim_sys   # averge over all realization //  offset -->  exp(s[0]) = tr(Identity)= 4 = dim_sys










########################################
# non-G classical saturation: K8c
########################################
 

y_A_qns = y_qns_A_K8 # matrix form // rows = exprmt , cols = window_index
y_B_qns = y_qns_B_K8

num_expr, num_window = np.shape(y_A_qns)[0:2]

read_rho = [sigma_33 for i in range(num_expr)] # should read from file // O ==ZZ is sufficient
init_list = init_list_K8 # QNS_rho



#all_meas_results_K4q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), num_expr))# to store all_meas_results
all_meas_results_K8 = np.zeros( (len(g_list),num_expr))

i1, i2,i3 = 0,0,0

#for Omega in Omega_list: # interate the cos_modultion in coupling 
 #trajectory_main = RTN_generator() 
trajectory_main = const_0mean_noise_generator()

#for TT_shift in Tshift_list: # interate the time_shift to quantum noise 
b_z = trajectory_main[:, 0:int(np.shape(trajectory_main)[1]*T/TIME_FINAL)+1]
b_x = np.zeros((b_z.shape[0],b_z.shape[1])) # a classical fluctuator only along z 
b_y = np.zeros((b_z.shape[0],b_z.shape[1]))

for g in g_list: # interate coupling strength
	g_A_x, g_A_y, g_A_z =  0, 0 , g # two qubits have same coupling 
	g_B_x, g_B_y, g_B_z =  0, 0 , g # two qubits have same coupling 
	obs_avg_simu = [] # [obs_x_avg(i) for i in range (6)]
	for i in range(num_expr):
		obs_avg_simu.append(obs_avg(sigma_33,i)) 
		#print("now:",datetime.now().strftime("%H:%M:%S"))
		print('finish C_i, i = ',i)
	
	obs_avg_simu=np.array(obs_avg_simu)    
	
	all_meas_results_K8[g_list.tolist().index(g)] = obs_avg_simu
	print("now:",datetime.now().strftime("%H:%M:%S"))
	#print('finish Omega,TT_shift,g ',Omega_list.tolist().index(Omega), Tshift_list.tolist().index(TT_shift), g_list.tolist().index(g))

#for g in g_list: # interate coupling strength
#g_A_x, g_A_y, g_A_z =  0, 0 , g_A
#g_B_x, g_B_y, g_B_z =  0, 0 , g_B
#obs_avg_simu = [] # [obs_x_avg(i) for i in range ()]
#for i in range(num_expr):
#	obs_avg_simu.append(obs_avg(sigma_33,i)) 
#	#print("now:",datetime.now().strftime("%H:%M:%S"))
#	print('finish C_i, i = ',i)

#obs_avg_simu=np.array(obs_avg_simu)    

#all_meas_results_K8 = obs_avg_simu
#print("now:",datetime.now().strftime("%H:%M:%S"))

np.save('all_meas_results_K8.npy',all_meas_results_K8)


























