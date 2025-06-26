import numpy as np
#from qutip import *
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
import sys
from scipy import interpolate
from scipy.integrate import quad, dblquad,nquad
from scipy.linalg import expm
#import matplotlib.pyplot as plt

import os
#os.chdir('/Users/wenzheng/Dropbox (Dartmouth College)/Code/check quantum noise')
#print("Current working directory: {0}".format(os.getcwd()))

from CA_params_setting import *

from C_qns_K4_L4_quantum import *
from C_qns_K4_L4_classical import *
from C_qns_K2_L4_quantum import *
from C_qns_K2_L4_classical import *

n_cores = 6 #


#######################################
# setups
#######################################

cos = np.cos
sin = np.sin
sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1= np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])
pauli_1_generator = [sigma_0,sigma_1,sigma_2,sigma_3]

L =4 # number of windows
dim_sys = 2 # dimensional of qubit-system


########################################
# Some essential functions
########################################

def RTN_generator():
	"""
	1. produce a zero mean noise
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


def U_sb(i,j, tt):
	"""
	i is C_ctrl index, j is realization index;
	tt is time , 0<=tt<= T
	"""
	# in the following, coeff_u_v is interp of  {y_u(t) * b_v(t)} 
	coeff_x_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_x[j,k] for k in range(num_steps)])  
	coeff_x_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_y[j,k] for k in range(num_steps)])
	coeff_x_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][0] * g_x*b_z[j,k] for k in range(num_steps)])
	coeff_y_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_x[j,k] for k in range(num_steps)])
	coeff_y_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_y[j,k] for k in range(num_steps)])
	coeff_y_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][1] * g_y*b_z[j,k] for k in range(num_steps)])
	coeff_z_x = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_x[j,k] for k in range(num_steps)])
	coeff_z_y = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_y[j,k] for k in range(num_steps)])    
	coeff_z_z = interpolate.interp1d(time_list,[y_qns[i,int(np.floor(k/(num_steps/L)))][2] * g_z*b_z[j,k] for k in range(num_steps)])
	steps = int(tt/T*(num_steps))   # if tt=T, steps = num_steps
	matrix = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	for k in range(num_steps):
		matrix = expm(-1j*(tt/steps)*((coeff_x_x(k/steps*tt)*np.kron(sigma_1,sigma_1)+
                                       coeff_x_y(k/steps*tt)*np.kron(sigma_1,sigma_2)+
                                       coeff_x_z(k/steps*tt)*np.kron(sigma_1,sigma_3)+
                                       coeff_y_x(k/steps*tt)*np.kron(sigma_2,sigma_1)+
                                       coeff_y_y(k/steps*tt)*np.kron(sigma_2,sigma_2)+
                                       coeff_y_z(k/steps*tt)*np.kron(sigma_2,sigma_3)+
                                       coeff_z_x(k/steps*tt)*np.kron(sigma_3,sigma_1)+
                                       coeff_z_y(k/steps*tt)*np.kron(sigma_3,sigma_2)+
                                       coeff_z_z(k/steps*tt)*np.kron(sigma_3,sigma_3))))@ matrix
	return matrix

def U_sb_dag(i,j,tt):
	return U_sb(i,j,tt).getH()

def obs_avg(O,i,rho_B=1/2* (sigma_0+sigma_3)):
	"""
	<O(T)>|_{rho_s,rho_b}
	where i tells which "ctrl" and which  qubit init_state
	rho_B is the intial_state of bath
	T will be fixed at final time
	"""
	b_dim=2
	f_map = lambda s: sigma_0 if (s==1) else (sigma_3 if s==2 else ( sigma_2 if s==3 else sigma_2) ) # map the inital state s==2 means rho_S =O = σ[z]
	exp_f =lambda j: np.trace(U_sb(i,j,T) @ np.kron(f_map(init_list[i]),rho_B) \
									@  U_sb_dag(i,j,T) @ np.kron(O, sigma_0) )  # fix the * -> @ bug
	obs_each = Parallel(n_jobs=n_cores, verbose=0)(delayed(exp_f)(j) for j in range (ENSEMBLE_SIZE) )
    
	return np.average(obs_each)-int(init_list[i]==2)*2   # # tr(I.rhoS.O)|_{rhoS==O} = tr(I)=2
 





########################################
# G classical: K2c
########################################
 

y_qns = y_qns_K2c # matrix form // rows = exprmt , cols = window_index 
num_expr, num_window = np.shape(y_qns)[0:2]

read_rho = [sigma_3 for i in range(num_expr)] # should read from file
init_list = init_list_K2c # QNS_rho



all_meas_results_K2c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), num_expr))# to store all_meas_results

i1, i2,i3 = 0,0,0

for Omega in Omega_list: # interate the cos_modultion in coupling 
	trajectory_main = RTN_generator() 

	for TT_shift in Tshift_list: # interate the time_shift to quantum noise 
		b_x = trajectory_main[:,0:int(np.shape(trajectory_main)[1]*T/TIME_FINAL)+1]
		b_y = trajectory_main[:,int(np.shape(trajectory_main)[1]*TT_shift/TIME_FINAL): int(np.shape(trajectory_main)[1]*(TT_shift+T)/TIME_FINAL)+5]
		b_y = b_y[0:b_x.shape[0],  0:b_x.shape[1]] # trim the 2nd-dim of b_y to match b_x
		b_z = np.zeros((b_x.shape[0],b_x.shape[1]))

		for g in g_list: # interate coupling strength
			g_x, g_y, g_z =  g, g , g
			obs_avg_simu = [] # [obs_x_avg(i) for i in range (6)]
			for i in range(num_expr):
 				obs_avg_simu.append(obs_avg(sigma_3,i)) 
 				#print("now:",datetime.now().strftime("%H:%M:%S"))
 				print('finish C_i, i = ',i)
			
			obs_avg_simu=np.array(obs_avg_simu)    
			
			all_meas_results_K2c[Omega_list.tolist().index(Omega)]\
			[Tshift_list.tolist().index(TT_shift)][g_list.tolist().index(g)] = obs_avg_simu

			print("now:",datetime.now().strftime("%H:%M:%S"))
			print('finish Omega,TT_shift,g ',Omega_list.tolist().index(Omega), Tshift_list.tolist().index(TT_shift), g_list.tolist().index(g))

np.save('all_meas_results_K2c.npy',all_meas_results_K2c)









########################################
# G quantum: K2q
########################################
 

y_qns = y_qns_K2q # matrix form // rows = exprmt , cols = window_index 
num_expr, num_window = np.shape(y_qns)[0:2]

read_rho = [sigma_3 for i in range(num_expr)] # should read from file
init_list = init_list_K2q # QNS_rho



all_meas_results_K2q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), num_expr))# to store all_meas_results

i1, i2,i3 = 0,0,0

for Omega in Omega_list: # interate the cos_modultion in coupling 
	trajectory_main = RTN_generator() 

	for TT_shift in Tshift_list: # interate the time_shift to quantum noise 
		b_x = trajectory_main[:,0:int(np.shape(trajectory_main)[1]*T/TIME_FINAL)+1]
		b_y = trajectory_main[:,int(np.shape(trajectory_main)[1]*TT_shift/TIME_FINAL): int(np.shape(trajectory_main)[1]*(TT_shift+T)/TIME_FINAL)+5]
		b_y = b_y[0:b_x.shape[0],  0:b_x.shape[1]] # trim the 2nd-dim of b_y to match b_x
		b_z = np.zeros((b_x.shape[0],b_x.shape[1]))

		for g in g_list: # interate coupling strength
			g_x, g_y, g_z =  g, g , g
			obs_avg_simu = [] # [obs_x_avg(i) for i in range (6)]
			for i in range(num_expr):
 				obs_avg_simu.append(obs_avg(sigma_3,i)) 
 				#print("now:",datetime.now().strftime("%H:%M:%S"))
 				print('finish C_i, i = ',i)
			
			obs_avg_simu=np.array(obs_avg_simu)    
			
			all_meas_results_K2q[Omega_list.tolist().index(Omega)]\
			[Tshift_list.tolist().index(TT_shift)][g_list.tolist().index(g)] = obs_avg_simu

			print("now:",datetime.now().strftime("%H:%M:%S"))
			print('finish Omega,TT_shift,g ',Omega_list.tolist().index(Omega), Tshift_list.tolist().index(TT_shift), g_list.tolist().index(g))

np.save('all_meas_results_K2q.npy',all_meas_results_K2q)









########################################
# non-G classical: K4c
########################################
 

y_qns = y_qns_K4c # matrix form // rows = exprmt , cols = window_index 
num_expr, num_window = np.shape(y_qns)[0:2]

read_rho = [sigma_3 for i in range(num_expr)] # should read from file
init_list = init_list_K4c # QNS_rho



all_meas_results_K4c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), num_expr)) # to store all_meas_results

i1, i2,i3 = 0,0,0

for Omega in Omega_list: # interate the cos_modultion in coupling 
	trajectory_main = RTN_generator() 

	for TT_shift in Tshift_list: # interate the time_shift to quantum noise 
		b_x = trajectory_main[:,0:int(np.shape(trajectory_main)[1]*T/TIME_FINAL)+1]
		b_y = trajectory_main[:,int(np.shape(trajectory_main)[1]*TT_shift/TIME_FINAL): int(np.shape(trajectory_main)[1]*(TT_shift+T)/TIME_FINAL)+5]
		b_y = b_y[0:b_x.shape[0],  0:b_x.shape[1]] # trim the 2nd-dim of b_y to match b_x
		b_z = np.zeros((b_x.shape[0],b_x.shape[1]))

		for g in g_list: # interate coupling strength
			g_x, g_y, g_z =  g, g , g
			obs_avg_simu = [] # [obs_x_avg(i) for i in range (6)]
			for i in range(num_expr):
 				obs_avg_simu.append(obs_avg(sigma_3,i)) 
 				#print("now:",datetime.now().strftime("%H:%M:%S"))
 				print('finish C_i, i = ',i)
			
			obs_avg_simu=np.array(obs_avg_simu)    
			
			all_meas_results_K4c[Omega_list.tolist().index(Omega)]\
			[Tshift_list.tolist().index(TT_shift)][g_list.tolist().index(g)] = obs_avg_simu

			print("now:",datetime.now().strftime("%H:%M:%S"))
			print('finish Omega,TT_shift,g ',Omega_list.tolist().index(Omega), Tshift_list.tolist().index(TT_shift), g_list.tolist().index(g))

np.save('all_meas_results_K4c.npy',all_meas_results_K4c)










########################################
# non-G quantum: K4q
########################################
 

y_qns = y_qns_K4q # matrix form // rows = exprmt , cols = window_index 
num_expr, num_window = np.shape(y_qns)[0:2]

read_rho = [sigma_3 for i in range(num_expr)] # should read from file
init_list = init_list_K4q # QNS_rho



all_meas_results_K4q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), num_expr))# to store all_meas_results

i1, i2,i3 = 0,0,0

for Omega in Omega_list: # interate the cos_modultion in coupling 
	trajectory_main = RTN_generator() 

	for TT_shift in Tshift_list: # interate the time_shift to quantum noise 
		b_x = trajectory_main[:,0:int(np.shape(trajectory_main)[1]*T/TIME_FINAL)+1]
		b_y = trajectory_main[:,int(np.shape(trajectory_main)[1]*TT_shift/TIME_FINAL): int(np.shape(trajectory_main)[1]*(TT_shift+T)/TIME_FINAL)+5]
		b_y = b_y[0:b_x.shape[0],  0:b_x.shape[1]] # trim the 2nd-dim of b_y to match b_x
		b_z = np.zeros((b_x.shape[0],b_x.shape[1]))

		for g in g_list: # interate coupling strength
			g_x, g_y, g_z =  g, g , g
			obs_avg_simu = [] # [obs_x_avg(i) for i in range (6)]
			for i in range(num_expr):
 				obs_avg_simu.append(obs_avg(sigma_3,i)) 
 				#print("now:",datetime.now().strftime("%H:%M:%S"))
 				print('finish C_i, i = ',i)
			
			obs_avg_simu=np.array(obs_avg_simu)    
			
			all_meas_results_K4q[Omega_list.tolist().index(Omega)]\
			[Tshift_list.tolist().index(TT_shift)][g_list.tolist().index(g)] = obs_avg_simu

			print("now:",datetime.now().strftime("%H:%M:%S"))
			print('finish Omega,TT_shift,g ',Omega_list.tolist().index(Omega), Tshift_list.tolist().index(TT_shift), g_list.tolist().index(g))

np.save('all_meas_results_K4q.npy',all_meas_results_K4q)










