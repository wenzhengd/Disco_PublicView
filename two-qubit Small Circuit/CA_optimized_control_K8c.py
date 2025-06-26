#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1, 2,...8
include the frame functions of 

K8c  [non-Gauss classical]
--------------------------------------------
see "2qubit_QNS_classical[CHN].nb"

"""

import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
#from qutip import * 
import pandas as pd 
from matplotlib import pyplot as plt
import scipy.optimize as opt
from scipy.linalg import expm 

import sys
from importlib import reload # https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module

#from CA_frame_functions import control_dynamics, sigma_x_T, sigma_y_T,sigma_z_T,S1_qns_data,S2_qns_data,S3_qns_data,S4_qns_data
from CA_params_setting import g_list #Omega_list, Tshift_list
from C_2qb_qns_K8_L2_classical import *

cos = np.cos
sin = np.sin
sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
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

L=2

n_cores = 6


############################################################
#    Read the QNS results and obtain spectra      
############################################################

obs_avg_simu = np.load('all_meas_results_K8.npy')
S_qns_results_K8c = np.zeros(np.array(obs_avg_simu).shape, dtype = 'complex_') # create array with same shape
optimize_K8c_sol = np.zeros((len(g_list), 30)) # we have 30 ctrl_params 

S1_qns_data_K8c = np.zeros((len(g_list),2,        L),dtype = 'complex_')#  qubit & window
S2_qns_data_K8c = np.zeros((len(g_list),2,2,      L,L),dtype = 'complex_') 
S3_qns_data_K8c = np.zeros((len(g_list),2,2,2,    L,L,L),dtype = 'complex_') 
S4_qns_data_K8c = np.zeros((len(g_list),2,2,2,2,  L,L,L,L),dtype = 'complex_')
S5_qns_data_K8c = np.zeros((len(g_list),2,2,2,2,2,  L,L,L,L,L),dtype = 'complex_')
S6_qns_data_K8c = np.zeros((len(g_list),2,2,2,2,2,2,  L,L,L,L,L,L),dtype = 'complex_')
S7_qns_data_K8c = np.zeros((len(g_list),2,2,2,2,2,2,2,  L,L,L,L,L,L,L),dtype = 'complex_')
S8_qns_data_K8c = np.zeros((len(g_list),2,2,2,2,2,2,2,2,  L,L,L,L,L,L,L,L),dtype = 'complex_')  

#procedure to have ALL spectra [for all configs, k-order, sign, window etc..]
# notice: the protocols are not in  dimensional form, they are specified in NAMES: Sk_qns_data_Kkc(q)
     #for i1 in range(len(Omega_list)):
	#for i2 in range(len(Tshift_list)):
for i3 in range(len(g_list)):
	S_qns_results_K8c = np.ndarray.flatten(O_to_S_matrix_K8 @ (np.matrix(obs_avg_simu[i3]).T) ) # read the obs result for Omega Tshift g
	[[S1_qns_data_K8c[i3][0][0] , S1_qns_data_K8c[i3][0][1] , S1_qns_data_K8c[i3][1][0] , S1_qns_data_K8c[i3][1][1] , \
	S2_qns_data_K8c[i3][0][0][0][0] , S2_qns_data_K8c[i3][0][0][1][0] , S2_qns_data_K8c[i3][0][0][1][1] , S2_qns_data_K8c[i3][0][1][0][0] ,  S2_qns_data_K8c[i3][0][1][1][0] ,\
	S2_qns_data_K8c[i3][0][1][1][1] , S2_qns_data_K8c[i3][1][0][1][0] , S2_qns_data_K8c[i3][1][1][0][0] , S2_qns_data_K8c[i3][1][1][1][0] , S2_qns_data_K8c[i3][1][1][1][1] ,\
	S3_qns_data_K8c[i3][0][0][0][1][0][0] , S3_qns_data_K8c[i3][0][0][0][1][1][0] , S3_qns_data_K8c[i3][0][0][1][0][0][0] , S3_qns_data_K8c[i3][0][0][1][1][0][0] , S3_qns_data_K8c[i3][0][0][1][1][1][0] , \
	S3_qns_data_K8c[i3][0][0][1][1][1][1] , S3_qns_data_K8c[i3][0][1][0][1][1][0] , S3_qns_data_K8c[i3][0][1][1][0][0][0] , S3_qns_data_K8c[i3][0][1][1][1][0][0] , S3_qns_data_K8c[i3][0][1][1][1][1][0] , \
	S3_qns_data_K8c[i3][0][1][1][1][1][1] , S3_qns_data_K8c[i3][1][0][0][1][0][0] , S3_qns_data_K8c[i3][1][0][1][1][0][0] , S3_qns_data_K8c[i3][1][1][0][1][1][0] , S3_qns_data_K8c[i3][1][1][1][1][0][0] , \
	S3_qns_data_K8c[i3][1][1][1][1][1][0] , S4_qns_data_K8c[i3][0][0][0][0][1][1][0][0] , S4_qns_data_K8c[i3][0][0][0][1][1][0][0][0] , S4_qns_data_K8c[i3][0][0][0][1][1][1][0][0] , \
	S4_qns_data_K8c[i3][0][0][1][0][1][1][1][0] , S4_qns_data_K8c[i3][0][0][1][1][0][0][0][0] , S4_qns_data_K8c[i3][0][0][1][1][1][0][0][0] , S4_qns_data_K8c[i3][0][0][1][1][1][1][0][0] , \
	S4_qns_data_K8c[i3][0][0][1][1][1][1][1][0] , S4_qns_data_K8c[i3][0][0][1][1][1][1][1][1] , S4_qns_data_K8c[i3][0][1][0][0][1][1][0][0] , S4_qns_data_K8c[i3][0][1][0][1][1][1][0][0] , \
	S4_qns_data_K8c[i3][0][1][1][0][1][1][1][0] , S4_qns_data_K8c[i3][0][1][1][1][1][1][0][0] , S4_qns_data_K8c[i3][0][1][1][1][1][1][1][0] , S4_qns_data_K8c[i3][1][0][0][1][1][0][0][0] , \
	S4_qns_data_K8c[i3][1][0][1][1][1][0][0][0] , S4_qns_data_K8c[i3][1][1][0][0][1][1][0][0] , S4_qns_data_K8c[i3][1][1][0][1][1][1][0][0] , S4_qns_data_K8c[i3][1][1][1][1][1][1][0][0] , \
	S5_qns_data_K8c[i3][0][0][0][0][1][1][1][0][0][0] , S5_qns_data_K8c[i3][0][0][0][1][1][1][0][0][0][0] , S5_qns_data_K8c[i3][0][0][0][1][1][1][1][0][0][0] , S5_qns_data_K8c[i3][0][0][1][0][0][1][1][1][0][0] , \
	S5_qns_data_K8c[i3][0][0][1][0][1][1][1][1][0][0] , S5_qns_data_K8c[i3][0][0][1][1][0][1][1][1][1][0] , S5_qns_data_K8c[i3][0][0][1][1][1][1][1][1][0][0] ,\
	S5_qns_data_K8c[i3][0][0][1][1][1][1][1][1][1][0] , S5_qns_data_K8c[i3][0][1][0][0][1][1][1][0][0][0] , S5_qns_data_K8c[i3][0][1][0][1][1][1][1][0][0][0] , S5_qns_data_K8c[i3][0][1][1][0][0][1][1][1][0][0] , \
	S5_qns_data_K8c[i3][0][1][1][0][1][1][1][1][0][0] , S5_qns_data_K8c[i3][0][1][1][1][1][1][1][1][0][0] , S5_qns_data_K8c[i3][1][0][0][1][1][1][0][0][0][0] ,\
	S5_qns_data_K8c[i3][1][1][0][0][1][1][1][0][0][0] , S5_qns_data_K8c[i3][1][1][0][1][1][1][1][0][0][0] , \
	S6_qns_data_K8c[i3][0][0][0][0][1][1][1][1][0][0][0][0] , S6_qns_data_K8c[i3][0][0][1][0][0][1][1][1][1][0][0][0] , S6_qns_data_K8c[i3][0][0][1][0][1][1][1][1][1][0][0][0] , S6_qns_data_K8c[i3][0][0][1][1][0][0][1][1][1][1][0][0] , \
	S6_qns_data_K8c[i3][0][0][1][1][0][1][1][1][1][1][0][0] , S6_qns_data_K8c[i3][0][0][1][1][1][1][1][1][1][1][0][0] , S6_qns_data_K8c[i3][0][1][0][0][1][1][1][1][0][0][0][0] , S6_qns_data_K8c[i3][0][1][1][0][0][1][1][1][1][0][0][0] , \
	S6_qns_data_K8c[i3][0][1][1][0][1][1][1][1][1][0][0][0] , S6_qns_data_K8c[i3][1][1][0][0][1][1][1][1][0][0][0][0] , \
	S7_qns_data_K8c[i3][0][0][1][0][0][1][1][1][1][1][0][0][0][0] , S7_qns_data_K8c[i3][0][0][1][1][0][0][1][1][1][1][1][0][0][0] , S7_qns_data_K8c[i3][0][0][1][1][0][1][1][1][1][1][1][0][0][0] , S7_qns_data_K8c[i3][0][1][1][0][0][1][1][1][1][1][0][0][0][0] , \
	S8_qns_data_K8c[i3][0][0][1][1][0][0][1][1][1][1][1][1][0][0][0][0] ]] = (S_qns_results_K8c.tolist())





############################################################
#    Essential functions of qubit dynamics: ctrl and noise      
############################################################

# Replace the Sn_vec_func() to S_qns results/knowledge

def S1_vec_fun(bath_params, q,n):
	"""
	bath_params = [g_index]   #[Omega_idx, Tshift_idx, g_idx]
	0<=q<=1  1<=n<=L // NOT use bound_error_check to SAVE time
	"""
	return S1_qns_data_K8c[bath_params[0]][q-1][n-1] 

def S2_vec_fun(bath_params, q1,q2, n1,n2):
	return S2_qns_data_K8c[bath_params[0]][q1-1][q2-1][n1-1][n2-1]

def S3_vec_fun(bath_params, q1,q2,q3, n1,n2,n3):
	return S3_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][n1-1][n2-1][n3-1]

def S4_vec_fun(bath_params , q1,q2,q3,q4, n1,n2,n3,n4):
	return S4_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][q4-1][n1-1][n2-1][n3-1][n4-1]		

def S5_vec_fun(bath_params , q1,q2,q3,q4,q5, n1,n2,n3,n4,n5):
	return S5_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][q4-1][q5-1][n1-1][n2-1][n3-1][n4-1][n5-1]

def S6_vec_fun(bath_params , q1,q2,q3,q4,q5,q6, n1,n2,n3,n4,n5,n6):
	return S6_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][q4-1][q5-1][q6-1][n1-1][n2-1][n3-1][n4-1][n5-1][n6-1]

def S7_vec_fun(bath_params , q1,q2,q3,q4,q5,q6,q7, n1,n2,n3,n4,n5,n6,n7):
	return S7_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][q4-1][q5-1][q6-1][q7-1][n1-1][n2-1][n3-1][n4-1][n5-1][n6-1][n7-1]

def S8_vec_fun(bath_params , q1,q2,q3,q4,q5,q6,q7,q8, n1,n2,n3,n4,n5,n6,n7,n8):
	return S8_qns_data_K8c[bath_params[0]][q1-1][q2-1][q3-1][q4-1][q5-1][q6-1][q7-1][q8-1][n1-1][n2-1][n3-1][n4-1][n5-1][n6-1][n7-1][n8-1] 



# Qubit control part //KAK params

def KAK(qubit_params, n):
	"""
	claculate the ***U_0(n)*** of ctrl Hamiltonian for given c_params
	-----------------------------------------------------------------
	qubit_params = [theta_1,.....,theta_15]
	-----------------------------------------------------------------
	theta = {0:theta_1A__1, 1:alpha_1A__1, 2:beta_1A__1, 3:theta_1B__1, 4:alpha_1B__1, 5:beta_1B__1, 6:THETA__1, 7:ALPHA__1, 8: BETA__1, 9: theta_2A__1, 10:alpha_2A__1, 11:beta_2A__1, 12: theta_2B__1, 13: alpha_2B__1, 14:beta_2B__1,\
	       15:theta_1A__2, 16:alpha_1A__2, 17:beta_1A__2, 18:theta_1B__2, 19:alpha_1B__2, 20:beta_1B__2, 21:THETA__2, 22:ALPHA__2, 23: BETA__2, 24: theta_2A__2, 25:alpha_2A__2, 26:beta_2A__2, 27: theta_2B__2, 28: alpha_2B__2, 29:beta_2B__2 }
	if n==1: np.kron(cos(theta_1A__1)*sigma_0-1j*sin(theta_1A__1)*(cos(alpha_1A__1)*cos(beta_1A__1)*sigma_1 + cos(alpha_1A__1)*sin(beta_1A__1)*sigma_2 + sin(alpha_1A__1)*sigma_3), sigma_0 ) @ np.kron( sigma_0, cos(theta_1B__1)*sigma_0-1j*sin(theta_1B__1)*(cos(alpha_1B__1)*cos(beta_1B__1)*sigma_1 + cos(alpha_1B__1)*sin(beta_1B__1)*sigma_2 + sin(alpha_1B__1)*sigma_3) ) \
			@ expm(-1j*(THETA__1*(cos(ALPHA__1)*cos(BETA__1)*sigma_11+cos(ALPHA__1)*sin(BETA__1)*sigma_22+sin(ALPHA__1)*sigma_33)))  \
			@ np.kron(cos(theta_2A__1)*sigma_0-1j*sin(theta_2A__1)*(cos(alpha_2A__1)*cos(beta_2A__1)*sigma_1 + cos(alpha_2A__1)*sin(beta_2A__1)*sigma_2 + sin(alpha_2A__1)*sigma_3), sigma_0 ) @ np.kron( sigma_0, cos(theta_2B__1)*sigma_0-1j*sin(theta_2B__1)*(cos(alpha_2B__1)*cos(beta_2B__1)*sigma_1 + cos(alpha_2B__1)*sin(beta_2B__1)*sigma_2 + sin(alpha_2B__1)*sigma_3) ) 
	if n==2:
		 np.kron(cos(theta_1A__2)*sigma_0-1j*sin(theta_1A__2)*(cos(alpha_1A__2)*cos(beta_1A__2)*sigma_1 + cos(alpha_1A__2)*sin(beta_1A__2)*sigma_2 + sin(alpha_1A__2)*sigma_3), sigma_0 ) @ np.kron( sigma_0, cos(theta_1B__2)*sigma_0-1j*sin(theta_1B__2)*(cos(alpha_1B__2)*cos(beta_1B__2)*sigma_1 + cos(alpha_1B__2)*sin(beta_1B__2)*sigma_2 + sin(alpha_1B__2)*sigma_3) ) \
			@ expm(-1j*(THETA__2*(cos(ALPHA__2)*cos(BETA__2)*sigma_11+cos(ALPHA__2)*sin(BETA__2)*sigma_22+sin(ALPHA__2)*sigma_33)))  @\
			np.kron(cos(theta_2A__2)*sigma_0-1j*sin(theta_2A__2)*(cos(alpha_2A__2)*cos(beta_2A__2)*sigma_1 + cos(alpha_2A__2)*sin(beta_2A__2)*sigma_2 + sin(alpha_2A__2)*sigma_3), sigma_0 ) @ np.kron( sigma_0, cos(theta_2B__2)*sigma_0-1j*sin(theta_2B__2)*(cos(alpha_2B__2)*cos(beta_2B__2)*sigma_1 + cos(alpha_2B__2)*sin(beta_2B__2)*sigma_2 + sin(alpha_2B__2)*sigma_3) ) 
	return        
	"""
	if n==1:
		return np.kron(cos(qubit_params[0])*sigma_0-1j*sin(qubit_params[0])*(cos(qubit_params[1])*cos(qubit_params[2])*sigma_1 \
			+ cos(qubit_params[1])*sin(qubit_params[2])*sigma_2 + sin(qubit_params[1])*sigma_3), sigma_0 ) \
			@ np.kron( sigma_0, cos(qubit_params[3])*sigma_0-1j*sin(qubit_params[3])*(cos(qubit_params[4])*cos(qubit_params[5])*sigma_1 \
			+ cos(qubit_params[4])*sin(qubit_params[5])*sigma_2 + sin(qubit_params[4])*sigma_3) ) \
			@ expm(-1j*(qubit_params[6]*(cos(qubit_params[7])*cos(qubit_params[8])*sigma_11+cos(qubit_params[7])*sin(qubit_params[8])*sigma_22+sin(qubit_params[7])*sigma_33)))  \
			@ np.kron(cos(qubit_params[9])*sigma_0-1j*sin(qubit_params[9])*(cos(qubit_params[10])*cos(qubit_params[11])*sigma_1 \
			+ cos(qubit_params[10])*sin(qubit_params[11])*sigma_2 + sin(qubit_params[10])*sigma_3), sigma_0 ) \
			@ np.kron( sigma_0, cos(qubit_params[12])*sigma_0-1j*sin(qubit_params[12])*(cos(qubit_params[13])*cos(qubit_params[14])*sigma_1 + cos(qubit_params[13])*sin(qubit_params[14])*sigma_2 + sin(qubit_params[13])*sigma_3) ) 
	else : # n==2
		return np.kron(cos(qubit_params[15])*sigma_0-1j*sin(qubit_params[15])*(cos(qubit_params[16])*cos(qubit_params[17])*sigma_1 \
			+ cos(qubit_params[16])*sin(qubit_params[17])*sigma_2 + sin(qubit_params[16])*sigma_3), sigma_0 ) \
			@ np.kron( sigma_0, cos(qubit_params[18])*sigma_0-1j*sin(qubit_params[18])*(cos(qubit_params[19])*cos(qubit_params[20])*sigma_1 \
				+ cos(qubit_params[19])*sin(qubit_params[20])*sigma_2 + sin(qubit_params[19])*sigma_3) ) \
			@ expm(-1j*(qubit_params[21]*(cos(qubit_params[22])*cos(qubit_params[23])*sigma_11+cos(qubit_params[22])*sin(qubit_params[23])*sigma_22+sin(qubit_params[22])*sigma_33))) \
			@ np.kron(cos(qubit_params[24])*sigma_0-1j*sin(qubit_params[24])*(cos(qubit_params[25])*cos(qubit_params[26])*sigma_1 + cos(qubit_params[25])*sin(qubit_params[26])*sigma_2 + sin(qubit_params[25])*sigma_3), sigma_0 ) \
			@ np.kron( sigma_0, cos(qubit_params[27])*sigma_0-1j*sin(qubit_params[27])*(cos(qubit_params[28])*cos(qubit_params[29])*sigma_1 + cos(qubit_params[28])*sin(qubit_params[29])*sigma_2 + sin(qubit_params[28])*sigma_3) ) 

def Y1(qubit_params, q,r,n):
	"""
	r is xi,xx,.....,zz (15) // q = 0 or 1 (A or B)
	"""	
	if q == 0: # q= A
		return np.trace( KAK(qubit_params,n).getH() @ sigma_31 @ KAK(qubit_params,n) @ pauli_2_generator[r])/4
	else:  # q==1 >>> q=B
		return  np.trace( KAK(qubit_params,n).getH() @ sigma_13 @ KAK(qubit_params,n) @ pauli_2_generator[r])/4

def ht(Obs,qubit_params,q,n):
	"""
	The H_tilde Hamiltonian on qubit 
	"""
	return sum(Y1(qubit_params,q,r,n)*pauli_2_generator[r] for r in range(1,1+15))

def hb(Obs,qubit_params,q,n):
	"""
	The H_bar Hamiltonian on qubit
	"""
	return - Obs @ ht(Obs,qubit_params,q,n) @ Obs ##  Obs.get() = Obs

def D0():
	return sigma_00

def D1(Obs,qubit_params,bath_params):
	"""
	Dyson-1 of V_O
	"""
	return -1j*sum( [(hb(Obs,qubit_params,q,n) * S1_vec_fun(bath_params,q,n) \
					+ ht(Obs,qubit_params,q,n)*  S1_vec_fun(bath_params,q,n) )  for q in range(0,2) for n in range(1,L+1)])

def D2(Obs,qubit_params,bath_params):
	"""
	Dyson-2 of V_O
	//////////
	The returned expr can be seen from Mathematica code on comple symbolic structure
	"""
	return sum([ \
		-1/2 * (hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) \
				+ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1)  \
				+ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) \
				+ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2)) * (S2_vec_fun(bath_params,q1,q2, n1,n2))  \
		if S2_vec_fun(bath_params,q1,q2, n1,n2) != 0 else 0 \
		for q1 in range(0,2) for q2 in range(0,2) for n1 in range(1,L+1) for n2 in range(1,n1+1)])

def D3(Obs,qubit_params,bath_params):
   """
   Dyson-3 of V_O
   """   
   return sum([ \
      1/4*1j* ( hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + \
                hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + \
                hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + \
                hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + \
                hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) + \
                hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) + \
                hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + \
                ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) )* S3_vec_fun(bath_params,q1,q2,q3, n1,n2,n3) \
      if S3_vec_fun(bath_params,q1,q2,q3, n1,n2,n3) != 0 else 0 \
      for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1)])

def D4(Obs,qubit_params,bath_params):
   """
   Dyson-4 of V_O
   """   
   return sum([ 1/8*\
      (	hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) +\
         hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) +\
         hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) +\
         hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) +\
         hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) +\
         ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) )* S4_vec_fun(bath_params,q1,q2,q3,q4, n1,n2,n3,n4)\
      if S4_vec_fun(bath_params,q1,q2,q3,q4,  n1,n2,n3,n4) != 0 else 0 \
      for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for q4 in range(0,2)   \
      for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1)])

def D5(Obs,qubit_params,bath_params):
	"""
	Dyson-5 of V_O
	"""	
	return sum([ -1/16*(1j)*\
       ( hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
   		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
   		hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) + 
   		ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5)) * S5_vec_fun(bath_params,q1,q2,q3,q4,q5,  n1,n2,n3,n4,n5)\
  		if S5_vec_fun(bath_params,q1,q2,q3,q4,q5,  n1,n2,n3,n4,n5) != 0 else 0 \
    	for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for q4 in range(0,2) for q5 in range(0,2) \
    	for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1) for n5 in range(1,n4+1)])    			

def D6(Obs,qubit_params,bath_params):
	"""
	Dyson-6 of V_O
	"""	
	return sum([ -1/32*\
     (hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
  		hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6)) * S6_vec_fun(bath_params,q1,q2,q3,q4,q5,q6,  n1,n2,n3,n4,n5,n6)\
 		if S6_vec_fun(bath_params,q1,q2,q3,q4,q5,q6, n1,n2,n3,n4,n5,n6) != 0 else 0 \
    	for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for q4 in range(0,2) for q5 in range(0,2) for q6 in range(0,2)	\
    	for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1) for n5 in range(1,n4+1) for n6 in range(1,n5+1)]) 

def D7(Obs,qubit_params,bath_params):
	"""
	Dyson-7 of V_O
	"""	
	return sum([ (1j)/64*\
      (hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q2,n2) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @    hb(Obs,qubit_params,q1,n1) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q1,n1) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @    ht(Obs,qubit_params,q2,n2) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @    ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @    ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @    ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @    ht(Obs,qubit_params,q6,n6) + 
  		 ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @    ht(Obs,qubit_params,q7,n7)) * S7_vec_fun(bath_params,q1,q2,q3,q4,q5,q6,q7,  n1,n2,n3,n4,n5,n6,n7)\
    	if S7_vec_fun(bath_params,q1,q2,q3,q4,q5,q6,q7,  n1,n2,n3,n4,n5,n6,n7) != 0 else 0 \
    	for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for q4 in range(0,2) for q5 in range(0,2) for q6 in range(0,2) for q7 in range(0,2)	\
    	for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1) for n5 in range(1,n4+1) for n6 in range(1,n5+1) for n7 in range(1,n6+1)]) 

def D8(Obs,qubit_params,bath_params):
	"""
	Dyson-8 of V_O
	"""	
	return sum([ 1/128*\
     ((hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q8,n8) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q7,n7) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @   hb(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @   hb(Obs,qubit_params,q2,n2) @ hb(Obs,qubit_params,q1,n1) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @   hb(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q1,n1) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ hb(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ hb(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q1,n1) @   ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ hb(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @   ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ hb(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @   ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) + 
  		 hb(Obs,qubit_params,q8,n8) @ hb(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @   ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) + 
  		 hb(Obs,qubit_params,q8,n8) @ ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @   ht(Obs,qubit_params,q6,n6) @ ht(Obs,qubit_params,q7,n7) + 
  		 ht(Obs,qubit_params,q1,n1) @ ht(Obs,qubit_params,q2,n2) @ ht(Obs,qubit_params,q3,n3) @ ht(Obs,qubit_params,q4,n4) @ ht(Obs,qubit_params,q5,n5) @ ht(Obs,qubit_params,q6,n6) @   ht(Obs,qubit_params,q7,n7) @ ht(Obs,qubit_params,q8,n8)))*\
  		  S8_vec_fun(bath_params,q1,q2,q3,q4,q5,q6,q7,q8,  n1,n2,n3,n4,n5,n6,n7,n8)\
  		if S8_vec_fun(bath_params,q1,q2,q3,q4,q5,q6,q7,q8,  n1,n2,n3,n4,n5,n6,n7,n8) != 0 else 0 \
    	for q1 in range(0,2) for q2 in range(0,2)   for q3 in range(0,2) for q4 in range(0,2) for q5 in range(0,2) for q6 in range(0,2) for q7 in range(0,2) for q8 in range(0,2)	\
    	for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1) for n5 in range(1,n4+1) for n6 in range(1,n5+1) for n7 in range(1,n6+1) for n8 in range(1,n7+1)]) 


# obtian the control dynamics:  add all Dyson together
def Dyson (Obs,qubit_params,bath_params):
	"""
	return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
	"""
	return D0() + D2(Obs,qubit_params,bath_params) # +D4(Obs,qubit_params,bath_params) \
				#+ D5(Obs,qubit_params,bath_params) + D6(Obs,qubit_params,bath_params) + D7(Obs,qubit_params,bath_params) + D8(Obs,qubit_params,bath_params)

def sigma_O_T(Obs,rhoS,qubit_params,bath_params):
	"""
	expectation value of pauli_2_matrix
	calculate the time-consumming dyson and store
	"""
	dyson = Dyson(Obs,qubit_params,bath_params)
	return  	 (np.array(dyson)[0,0]*(rhoS @ Obs)[0,0] + np.array(dyson)[0,1]*(rhoS @ Obs)[1,0] + np.array(dyson)[0,2]*(rhoS @ Obs)[2,0] + np.array(dyson)[0,3]*(rhoS @ Obs)[3,0]\
			+ np.array(dyson)[1,0]*(rhoS @ Obs)[0,1] + np.array(dyson)[1,1]*(rhoS @ Obs)[1,1] + np.array(dyson)[1,2]*(rhoS @ Obs)[2,1] + np.array(dyson)[1,3]*(rhoS @ Obs)[3,1]\
			+ np.array(dyson)[2,0]*(rhoS @ Obs)[0,2] + np.array(dyson)[2,1]*(rhoS @ Obs)[1,2] + np.array(dyson)[2,2]*(rhoS @ Obs)[2,2] + np.array(dyson)[2,3]*(rhoS @ Obs)[3,2]\
			+ np.array(dyson)[3,0]*(rhoS @ Obs)[0,3] + np.array(dyson)[3,1]*(rhoS @ Obs)[1,3] + np.array(dyson)[3,2]*(rhoS @ Obs)[2,3] + np.array(dyson)[3,3]*(rhoS @ Obs)[3,3])
		# this is Tr[D.rhoS.O] ; for noiseless D=id, <O> = 4 iff rhoS=O, =0 otherwise

def devi_sigma_O_T(Obs,rhoS,qubit_params,bath_params):
	"""
	deviation of sigma_O_T
	"""
	return abs(sigma_O_T(Obs,rhoS,qubit_params,bath_params)-4) if (Obs ==rhoS).all() else abs(sigma_O_T(Obs,rhoS,qubit_params,bath_params)) 
	#BC for noiseless D=id, <O> = 4 iff rhoS=O, =0 otherwise



############################################################
#    functions used in gate optimization      
############################################################

"""
we know that the fidleity=1/15* sum^15_{s=1} <O_s>_|rho_s)

"""
def infidelity(qubit_params,bath_params):
	"""
	"""
	#  change  the GLOBALparameters using the ctro_parames
	_O_ =[sigma_11, sigma_12, sigma_13, sigma_21, sigma_22,sigma_23, sigma_31, sigma_32, sigma_33]
	# due to dyson is SLOW to compute, use PARALLEL method
	#return 1-1/15*( sum( abs(sigma_O_T(o,o,qubit_params,bath_params)) for o in _O_) )
	return sum(Parallel(n_jobs=n_cores, verbose=0)(delayed(devi_sigma_O_T)(o,o,qubit_params,bath_params)  for  o in  _O_ ))

def l2_cost(ctrl_params,bath_params):
	return 0


def fidelity (qubit_params,bath_params):
	"""
	"""
	_O_ =[sigma_11, sigma_12, sigma_13, sigma_21, sigma_22,sigma_23, sigma_31, sigma_32, sigma_33]
	return 1-1/len(_O_)* sum(np.sqrt(devi_sigma_O_T(o,o,qubit_params,bath_params)) for o in  _O_ )





############################################################
#    K8c Gaussian classical optimization: devide & conquer
############################################################


optimize_K8c_sol_dac=np.ones((len(g_list),30))


# ~~~~~~~~~~~~~~   dac stage 1  ~~~~~~~~~~~~~~

L=1 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
initial_guess = [0]*15 # params in 1st bin
def f_opt_K8c_dac1(qubit_params_1,bath_params): #only focus on 1st 15(L=1) to optimize 
	return infidelity(np.concatenate((np.array(qubit_params_1),np.array([0]*15))) ,bath_params)

print("start opt L1:",datetime.now().strftime("%H:%M:%S"))
for  i3 in range(len(g_list)):
	_opt_ = opt.minimize(fun = f_opt_K8c_dac1, x0= initial_guess,args = ([i3]), method ='Nelder-Mead', options={'maxiter': 10000}) # args [i3] to make it list  <-> bath_parms
	optimize_K8c_sol_dac[i3][0:15] =  _opt_.x #
	initial_guess = _opt_.x
	print("finish i3=", i3, '@', datetime.now().strftime("%H:%M:%S"))
	print(np.array(_opt_.x))
print("End  opt L1:: ",datetime.now().strftime("%H:%M:%S"))



# ~~~~~~~~~~~~~~  dac stage 2  ~~~~~~~~~~~~~~

L=2 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
initial_guess = [0]*15 # params in 2nd bin
def f_opt_K8c_dac2(opt_params_2,bath_params): #only focus on 1st 15(L=1) to optimizz
	return infidelity(np.concatenate((optimize_K8c_sol_dac[ bath_params[0] ][0:15],np.array( opt_params_2))) ,bath_params) # plugin the fixed 1st 15_params @ L=1 optimization

print("start opt L2:",datetime.now().strftime("%H:%M:%S"))
for  i3 in range(len(g_list)):
	_opt_ = opt.minimize(fun = f_opt_K8c_dac2, x0= initial_guess,args = ([i3]), method ='Nelder-Mead',options={'maxiter': 10000})
	optimize_K8c_sol_dac[i3][15:] =  _opt_.x #
	initial_guess = _opt_.x
	print("finish i3=", i3, '@', datetime.now().strftime("%H:%M:%S"))
	print(np.array(_opt_.x))
print("End  opt L2: ",datetime.now().strftime("%H:%M:%S"))

np.save('opt_2qb_result_K8_dac.npy',optimize_K8c_sol_dac) 






#########################################################
#  C_raw fidleity  by plug back
#########################################################

np.save('opt_2qb_fidelity_C_opt.npy', np.array([fidelity(optimize_K8c_sol_dac[i3],[i3]) for i3 in range(len(g_list))])) #F(C_opt)
np.save('opt_2qb_fidelity_C_raw.npy', np.array([fidelity([0]*30,[i3]) for i3 in range(len(g_list))]))      # F(C_raw) where C_raw is no control





############################################################
#    K8c Gaussian classical optimization: AIO
############################################################


#def f_opt_K8c_AIO(i3 ):
#	"""
#    i3 are index of g_list
#	return is the optized control thetas that min the  "l2_cost"
#	"""	
#	# min l2_cost
#	#initial_guess = [0]*30
#	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = ([i3]), method ='Nelder-Mead')
#	return _opt_.x
#
#
#initial_guess = [0]*30
#optimize_K8c_sol=np.ones((len(g_list),30))
#print("start opt all K8:",datetime.now().strftime("%H:%M:%S"))
#for  i3 in range(len(g_list)):
#	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = ([i3]), method ='Nelder-Mead')
#	optimize_K8c_sol[i3] =  _opt_.x #
#	initial_guess = _opt_.x
#	print("finish i3=", i3, ' @ ', datetime.now().strftime("%H:%M:%S"))
#	print(np.array(_opt_.x))
#print("End  optimization: ",datetime.now().strftime("%H:%M:%S"))
#np.save('opt_2qb_result_K8.npy',optimize_K8c_sol)    





###########################################################
#    others 
############################################################


#print("start opt all K8:",datetime.now().strftime("%H:%M:%S"))
#optimize_K8c_sol=Parallel(n_jobs=n_cores, verbose=5)(delayed(f_opt_K8c_AIO)(i3)  for  i3 in range(len(g_list)))# range(len(g_list))
#optimize_K8c_sol = np.array(optimize_K8c_sol).reshape(len(g_list), 30) # The last 30=  we have 30 ctrl_params  
#np.save('opt_2qb_result_K8.npy',optimize_K8c_sol)
#print("end opt all K8:",datetime.now().strftime("%H:%M:%S"))


#Nfeval =1 #https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
#def callbackF(x,y):
#	global Nfeval
#	print('{0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f} {6: 3.6f} {7: 3.6f} {8: 3.6f} {9: 3.6f} {10: 3.6f} {11: 3.6f}{12: 3.6f}{13: 3.6f}{14: 3.6f}{15: 3.6f}{16: 3.6f}{17: 3.6f} '.\
#	format(Nfeval, x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14], infidelity(x,y)))
#	Nfeval += 1
#
#initial_guess = [0]*30
#print("Start  optimization: ",datetime.now().strftime("%H:%M:%S"))
#_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = ([0]), method ='Nelder-Mead')
#print("End  optimization: ",datetime.now().strftime("%H:%M:%S"))






