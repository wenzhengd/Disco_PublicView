#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=4  Dyson 1, 2, 3,4 
include the frame functions of 

K2c  [Gaussian classical]
K2q
K4c
K4q  [non-Gauss quantum]

"""

import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
#from qutip import * 
#import pandas as pd 
#from matplotlib import pyplot as plt
import scipy.optimize as opt

import sys
from importlib import reload # https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module

#from CA_frame_functions import control_dynamics, sigma_x_T, sigma_y_T,sigma_z_T,S1_qns_data,S2_qns_data,S3_qns_data,S4_qns_data
from CA_params_setting import Omega_list, Tshift_list, g_list


cos = np.cos
sin = np.sin
sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])
_O_ =[sigma_0,sigma_1,sigma_2,sigma_3]

L=4


n_cores = 20
MAX_ITER = 10000


############################################################
#    Essential functions of qubit dynamics: ctrl and noise      
############################################################

# Replace the Sn_vec_func() to S_qns results/knowledge
def S1_vec_fun(bath_params , n):
	"""
	bath_params: indicate the qns_protocol[0] and the params of g,Omg,TT [1,2,3], 
	e.g, bath_params = ['K2c',0,0,0] is 'K2c'-protocol-reconstructed spectra, Omega[0], Tshift[0], g[0]
	The S1_qns_data_K2c, say, is 4-dim, with each dim being g, Omg, TT, n
	"""
	if bath_params[0] == 'K2c':
		return S1_qns_data_K2c[bath_params[1]][bath_params[2]][bath_params[3]][n-1] if (1<=n<=4) else print("S1 bound error")
	elif bath_params[0] == 'K2q':
		return S1_qns_data_K2q[bath_params[1]][bath_params[2]][bath_params[3]][n-1] if (1<=n<=4) else print("S1 bound error")
	elif bath_params[0] == 'K4c':
		return S1_qns_data_K4c[bath_params[1]][bath_params[2]][bath_params[3]][n-1] if (1<=n<=4) else print("S1 bound error")
	elif bath_params[0] == 'K4q':
		return S1_qns_data_K4q[bath_params[1]][bath_params[2]][bath_params[3]][n-1] if (1<=n<=4) else print("S1 bound error")
	else:
		return null 

def S2_vec_fun(bath_params, mu, n1,n2):
	if bath_params[0] == 'K2c':
		return S2_qns_data_K2c[bath_params[1]][bath_params[2]][bath_params[3]][mu][n1-1][n2-1] if (0<=mu<=1) & (1<=n1<=4) & (1<=n2<=4) else print("S2 bound error")
	elif bath_params[0] == 'K2q':
		return S2_qns_data_K2q[bath_params[1]][bath_params[2]][bath_params[3]][mu][n1-1][n2-1] if (0<=mu<=1) & (1<=n1<=4) & (1<=n2<=4) else print("S2 bound error")
	elif bath_params[0] == 'K4c':
		return S2_qns_data_K4c[bath_params[1]][bath_params[2]][bath_params[3]][mu][n1-1][n2-1] if (0<=mu<=1) & (1<=n1<=4) & (1<=n2<=4) else print("S2 bound error")
	elif bath_params[0] == 'K4q':
		return S2_qns_data_K4q[bath_params[1]][bath_params[2]][bath_params[3]][mu][n1-1][n2-1] if (0<=mu<=1) & (1<=n1<=4) & (1<=n2<=4) else print("S2 bound error")
	else:
		return null 

def S3_vec_fun(bath_params, mu1,mu2,  n1,n2,n3):
	if bath_params[0] == 'K2c':
		return S3_qns_data_K2c[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][n1-1][n2-1][n3-1] if (0<=mu1<=1) &(0<=mu2<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) else print("S3 bound error")
	elif bath_params[0] == 'K2q':
		return S3_qns_data_K2q[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][n1-1][n2-1][n3-1] if (0<=mu1<=1) &(0<=mu2<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) else print("S3 bound error")
	elif bath_params[0] == 'K4c':
		return S3_qns_data_K4c[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][n1-1][n2-1][n3-1] if (0<=mu1<=1) &(0<=mu2<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) else print("S3 bound error")
	elif bath_params[0] == 'K4q':
		return S3_qns_data_K4q[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][n1-1][n2-1][n3-1] if (0<=mu1<=1) &(0<=mu2<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) else print("S3 bound error")
	else: 
		return null 

def S4_vec_fun(bath_params, mu1,mu2,mu3,  n1,n2,n3,n4):
	if bath_params[0] == 'K2c':
		return S4_qns_data_K2c[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][mu3][n1-1][n2-1][n3-1][n4-1] if \
			(0<=mu1<=1) & (0<=mu2<=1) & (0<=mu3<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) & (1<=n4<=4) else print("S4 bound error")
	elif bath_params[0] == 'K2q':
		return S4_qns_data_K2q[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][mu3][n1-1][n2-1][n3-1][n4-1] if \
			(0<=mu1<=1) & (0<=mu2<=1) & (0<=mu3<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) & (1<=n4<=4) else print("S4 bound error")
	elif bath_params[0] == 'K4c':
		return S4_qns_data_K4c[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][mu3][n1-1][n2-1][n3-1][n4-1] if \
			(0<=mu1<=1) & (0<=mu2<=1) & (0<=mu3<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) & (1<=n4<=4) else print("S4 bound error")
	elif bath_params[0] == 'K4q':
		return S4_qns_data_K4q[bath_params[1]][bath_params[2]][bath_params[3]][mu1][mu2][mu3][n1-1][n2-1][n3-1][n4-1] if \
			(0<=mu1<=1) & (0<=mu2<=1) & (0<=mu3<=1) & (1<=n1<=4) & (1<=n2<=4) & (1<=n3<=4) & (1<=n4<=4) else print("S4 bound error")
	else: 
		return null 	

# Qubit control part

def Y1(qubit_params, r,n):
	"""
	r is x,y,z ; qubit_params is a len=8 list,
	"""
	return sin(qubit_params[0+2*(n-1)])*cos(qubit_params[1+2*(n-1)]) if r==1 \
	  else sin(qubit_params[0+2*(n-1)])*sin(qubit_params[1+2*(n-1)]) if r==2 \
	  else cos(qubit_params[0+2*(n-1)])



def ht(Obs,qubit_params,n):
	"""
	The H_tilde Hamiltonian on qubit
	"""
	return Y1(qubit_params,1,n)*sigma_1+ Y1(qubit_params,2,n)*sigma_2+Y1(qubit_params,3,n)*sigma_3
def hb(Obs,qubit_params,n):
	"""
	The H_bar Hamiltonian on qubit
	"""
	return - Obs @ ht(Obs,qubit_params,n) @ Obs ##  Obs.get() = Obs

def	D0():
	"""
	Dyson-1 of V_O
	"""
	return sigma_0

def	D1(Obs,qubit_params,bath_params):
	"""
	Dyson-1 of V_O
	"""
	return -1j*sum( [(hb(Obs,qubit_params,n1) * S1_vec_fun(bath_params,n1) + ht(Obs,qubit_params,n1)*  S1_vec_fun(bath_params,n1) ) for n1 in range(1,L+1)])

def	D2(Obs,qubit_params,bath_params):
	"""
	Dyson-2 of V_O
	//////////
	The returned expr can be seen from Mathematica code on comple symbolic structure
	"""
	return sum([ \
		-1/2 * hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1) * (S2_vec_fun(bath_params,0,n1,n2) - S2_vec_fun(bath_params,1,n1,n2)) if (S2_vec_fun(bath_params,0,n1,n2) - S2_vec_fun(bath_params,1,n1,n2)) != 0 else 0 \
		-1/2 * hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1) * (S2_vec_fun(bath_params,0,n1,n2) - S2_vec_fun(bath_params,1,n1,n2)) if (S2_vec_fun(bath_params,0,n1,n2) - S2_vec_fun(bath_params,1,n1,n2)) != 0 else 0 \
		-1/2 * hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) * (S2_vec_fun(bath_params,0,n1,n2) + S2_vec_fun(bath_params,1,n1,n2)) if (S2_vec_fun(bath_params,0,n1,n2) + S2_vec_fun(bath_params,1,n1,n2)) != 0 else 0 \
		-1/2 * ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) * (S2_vec_fun(bath_params,0,n1,n2) + S2_vec_fun(bath_params,1,n1,n2)) if (S2_vec_fun(bath_params,0,n1,n2) + S2_vec_fun(bath_params,1,n1,n2)) != 0 else 0  \
		for n1 in range(1,L+1) for n2 in range(1,n1+1)])

def	D3(Obs,qubit_params,bath_params):
	"""
	Dyson-3 of V_O
	"""	
	return sum([ \
		1/4*1j*(  (hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3)+hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3)) * (S3_vec_fun(bath_params,0,0, n1,n2,n3) + S3_vec_fun(bath_params,0,1, n1,n2,n3) - S3_vec_fun(bath_params,1,0, n1,n2,n3) - S3_vec_fun(bath_params,1,1, n1,n2,n3)) if (S3_vec_fun(bath_params,0,0, n1,n2,n3) + S3_vec_fun(bath_params,0,1, n1,n2,n3) - S3_vec_fun(bath_params,1,0, n1,n2,n3) - S3_vec_fun(bath_params,1,1, n1,n2,n3)) != 0 else 0  + \
   				(hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2)+hb(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2)) * (S3_vec_fun(bath_params,0,0, n1,n2,n3) - S3_vec_fun(bath_params,0,1, n1,n2,n3) + S3_vec_fun(bath_params,1,0, n1,n2,n3) - S3_vec_fun(bath_params,1,1, n1,n2,n3)) if (S3_vec_fun(bath_params,0,0, n1,n2,n3) - S3_vec_fun(bath_params,0,1, n1,n2,n3) + S3_vec_fun(bath_params,1,0, n1,n2,n3) - S3_vec_fun(bath_params,1,1, n1,n2,n3)) != 0 else 0  + \
   				(hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1)+hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1)) * (S3_vec_fun(bath_params,0,0, n1,n2,n3) - S3_vec_fun(bath_params,0,1, n1,n2,n3) - S3_vec_fun(bath_params,1,0, n1,n2,n3) + S3_vec_fun(bath_params,1,1, n1,n2,n3)) if (S3_vec_fun(bath_params,0,0, n1,n2,n3) - S3_vec_fun(bath_params,0,1, n1,n2,n3) - S3_vec_fun(bath_params,1,0, n1,n2,n3) + S3_vec_fun(bath_params,1,1, n1,n2,n3)) != 0 else 0  + \
   				(hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3)+ht(n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3)) * (S3_vec_fun(bath_params,0,0, n1,n2,n3) + S3_vec_fun(bath_params,0,1, n1,n2,n3) + S3_vec_fun(bath_params,1,0, n1,n2,n3) + S3_vec_fun(bath_params,1,1, n1,n2,n3)) if (S3_vec_fun(bath_params,0,0, n1,n2,n3) + S3_vec_fun(bath_params,0,1, n1,n2,n3) + S3_vec_fun(bath_params,1,0, n1,n2,n3) + S3_vec_fun(bath_params,1,1, n1,n2,n3)) != 0 else 0   ) \
		for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1)])

def D4(Obs,qubit_params,bath_params):
	"""
	Dyson-4 of V_O
	"""	
	return sum([ \
    1/8*(hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  + \
         hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n4) @ hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n3)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n3) @ hb(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) - S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         hb(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  +\
         ht(Obs,qubit_params,n1) @ ht(Obs,qubit_params,n2) @ ht(Obs,qubit_params,n3) @ ht(Obs,qubit_params,n4)*(S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4)) if (S4_vec_fun(bath_params,0, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,0, 1, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 0, 1, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 0, n1,n2,n3,n4) + S4_vec_fun(bath_params,1, 1, 1, n1,n2,n3,n4))  != 0 else 0  )\
    		for n1 in range(1,L+1) for n2 in range(1,n1+1) for n3 in range(1,n2+1) for n4 in range(1,n3+1)])	

# obtian the control dynamics: sympy add all Dyson together
def Dyson (Obs,qubit_params,bath_params):
    """
    ###COMMENT OUT###: It is actually dyson expansion with control -"i"- UNSPECIFIED!!!	
    return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
    """
    return D0()+D1(Obs,qubit_params,bath_params) + D2(Obs,qubit_params,bath_params) + D3(Obs,qubit_params,bath_params) + D4(Obs,qubit_params,bath_params)

#def sigma_x_T (qubit_params, rhoS, bath_params):
#    """
#    make the expectation value of pauli matrix
#    """
#    dyson = Dyson(sigma_1,qubit_params,bath_params)
#    expectation = (np.array(dyson)[0,0]*(rhoS @ sigma_1)[0,0]+ np.array(dyson)[1,0]*(rhoS @ sigma_1)[0,1]+ np.array(dyson)[0,1]*(rhoS @ sigma_1)[1,0]+ np.array(dyson)[1,1]*(rhoS @ sigma_1)[1,1]  )
#    # this is Tr[D.rho.O] for noiseless D=id, <O> =dim_sys
#    return expectation
#
#def sigma_y_T (qubit_params, rhoS, bath_params):
#    """
#    make the expectation value of pauli matrix
#    """
#    dyson = Dyson(sigma_2,qubit_params,bath_params)    
#    expectation = (np.array(dyson)[0,0]*(rhoS @ sigma_1)[0,0]+ np.array(dyson)[1,0]*(rhoS @ sigma_1)[0,1]+ np.array(dyson)[0,1]*(rhoS @ sigma_1)[1,0]+ np.array(dyson)[1,1]*(rhoS @ sigma_1)[1,1] )   
#     # this is Tr[D.rho.O] for noiseless D=id, <O> =dim_sys
#    return expectation
#
#def sigma_z_T (qubit_params, rhoS, bath_params):
#    """
#    make the expectation value of pauli matrix
#    """
#    dyson = Dyson(sigma_3,qubit_params,bath_params)    
#    expectation =1/2*( np.array(dyson)[0,0]*(rhoS @ sigma_1)[0,0]+ np.array(dyson)[1,0]*(rhoS @ sigma_1)[0,1]+ np.array(dyson)[0,1]*(rhoS @ sigma_1)[1,0]+ np.array(dyson)[1,1]*(rhoS @ sigma_1)[1,1])   
#     # this is Tr[D.rho.O] for noiseless D=id, <O> =dim_sys
#    return expectation

def sigma_O_T(Obs,rhoS,qubit_params,bath_params):
	"""
	expectation value of pauli_2_matrix
	calculate the time-consumming dyson and store
	"""
	dyson = Dyson(Obs,qubit_params,bath_params)
	return  	 (np.array(dyson)[0,0]*(rhoS @ Obs)[0,0] + np.array(dyson)[0,1]*(rhoS @ Obs)[1,0] \
			+ np.array(dyson)[1,0]*(rhoS @ Obs)[0,1] + np.array(dyson)[1,1]*(rhoS @ Obs)[1,1] )
		# this is Tr[D.rhoS.O] ; for noiseless D=id, <O> = 4 iff rhoS=O, =0 otherwise

def devi_sigma_O_T(Obs,rhoS,qubit_params,bath_params):
	"""
	deviation of sigma_O_T
	"""
	return abs(sigma_O_T(Obs,rhoS,qubit_params,bath_params)-2) if (Obs ==rhoS).all() else abs(sigma_O_T(Obs,rhoS,qubit_params,bath_params)) 
	#BC for noiseless D=id, <O> = 2 iff rhoS=O, =0 otherwise


############################################################
#    functions used in gate optimization      
############################################################

"""
we know that the fidleity=1/8(2+ sum^3_{s=1} <O_s>_|rho_s)

"""
def infidelity(qubit_params,bath_params):
	"""
	"""
	#  change  the GLOBALparameters using the ctro_parames
	# due to dyson is SLOW to compute, use PARALLEL method
	return sum(Parallel(n_jobs=n_cores, verbose=0)(delayed(devi_sigma_O_T)(o,o,qubit_params,bath_params)  for  o in  _O_ ))

def l2_cost(ctrl_params,bath_params):
	return 0
	# 	  	





############################################################
#    K2c Gaussian classical optimization 
############################################################
from C_qns_K2_L4_classical import * 
obs_avg_simu = np.load('all_meas_results_K2c.npy')
S_qns_results_K2c = np.zeros(np.array(obs_avg_simu).shape, dtype = 'complex_') # create array with same shape
optimize_K2c_sol = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), 8)) # we have 8 ctrl_params

S1_qns_data_K2c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),          4),dtype = 'complex_')# 
S2_qns_data_K2c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,      4,4),dtype = 'complex_')
S3_qns_data_K2c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,    4,4,4),dtype = 'complex_')
S4_qns_data_K2c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,2,  4,4,4,4),dtype = 'complex_') 

#procedure to have ALL spectra [for all configs, k-order, sign, window etc..]
# notice: the protocols are not in  dimensional form, they are specified in NAMES: Sk_qns_data_Kkc(q)
for i1 in range(len(Omega_list)):
	for i2 in range(len(Tshift_list)):
		for i3 in range(len(g_list)):
			S_qns_results_K2c[i1][i2][i3] = np.ndarray.flatten(O_to_S_matrix_K2c @ (np.matrix(obs_avg_simu[i1][i2][i3]).T) ) # read the obs result for Omega Tshift g
			[S1_qns_data_K2c[i1][i2][i3][0], S1_qns_data_K2c[i1][i2][i3][1], S1_qns_data_K2c[i1][i2][i3][2], S1_qns_data_K2c[i1][i2][i3][3], S2_qns_data_K2c[i1][i2][i3][0][0][0], \
			S2_qns_data_K2c[i1][i2][i3][0][1][0], S2_qns_data_K2c[i1][i2][i3][0][1][1], S2_qns_data_K2c[i1][i2][i3][0][2][0], S2_qns_data_K2c[i1][i2][i3][0][2][1],\
			S2_qns_data_K2c[i1][i2][i3][0][2][2], S2_qns_data_K2c[i1][i2][i3][0][3][0], S2_qns_data_K2c[i1][i2][i3][0][3][1], \
			S2_qns_data_K2c[i1][i2][i3][0][3][2], S2_qns_data_K2c[i1][i2][i3][0][3][3]]  = (S_qns_results_K2c[i1][i2][i3].tolist())


def f_opt_K2c_AIO(i1,i2,i3,guess=[0,np.pi,0,np.pi,1,1,1,1]):
	"""
	i1,i2,i3 are index of Omega_list,Tshift_list,g_list respectively 
	return is the optized control theta[1:4] and phi[1:4] that min the  "l2_cost"
	"""
	# min l2_cost
	initial_guess = guess
	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = (['K2c',i1,i2,i3]), method ='Nelder-Mead',options={'maxiter': MAX_ITER})
	return  _opt_.x

#f_opt_K2c_AIO(0,0,0)
print("start opt all K2c:",datetime.now().strftime("%H:%M:%S"))
optimize_K2c_sol = Parallel(n_jobs=n_cores, verbose=0)(delayed(f_opt_K2c_AIO)(i1,i2,i3) for i1 in range(len(Omega_list)) \
	for  i2 in range(len(Tshift_list)) for  i3 in range(len(g_list)))
optimize_K2c_sol = np.array(optimize_K2c_sol).reshape(len(Omega_list),len(Tshift_list),len(g_list), 8) # The last 8=  we have 8 ctrl_params  
np.save('opt_result_K2c.npy',optimize_K2c_sol)
print("end opt all K2c:",datetime.now().strftime("%H:%M:%S"))








############################################################
#    K2q non-G classical optimization 
############################################################
from C_qns_K2_L4_quantum import * 
obs_avg_simu = np.load('all_meas_results_K2q.npy')
S_qns_results_K2q = np.zeros(np.array(obs_avg_simu).shape ,dtype = 'complex_') # create array with same shape
optimize_K2q_sol = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), 8)) # we have 8 ctrl_params

S1_qns_data_K2q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),          4),dtype = 'complex_')# 
S2_qns_data_K2q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,      4,4),dtype = 'complex_')
S3_qns_data_K2q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,    4,4,4),dtype = 'complex_')
S4_qns_data_K2q = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,2,  4,4,4,4),dtype = 'complex_') 

#procedure to have ALL spectra [for all configs, k-order, sign, window etc..]
# notice: the protocols are not in  dimensional form, they are specified in NAMES: Sk_qns_data_Kkc(q)
for i1 in range(len(Omega_list)):
	for i2 in range(len(Tshift_list)):
		for i3 in range(len(g_list)):
			S_qns_results_K2q[i1][i2][i3] = np.ndarray.flatten(O_to_S_matrix_K2q @ (np.matrix(obs_avg_simu[i1][i2][i3]).T) ) # read the obs result for Omega Tshift g
			[S1_qns_data_K2q[i1][i2][i3][0], S1_qns_data_K2q[i1][i2][i3][1], S1_qns_data_K2q[i1][i2][i3][2], S1_qns_data_K2q[i1][i2][i3][3], S2_qns_data_K2q[i1][i2][i3][0][0][0], S2_qns_data_K2q[i1][i2][i3][0][1][0], S2_qns_data_K2q[i1][i2][i3][0][1][1], \
			S2_qns_data_K2q[i1][i2][i3][0][2][0], S2_qns_data_K2q[i1][i2][i3][0][2][1], S2_qns_data_K2q[i1][i2][i3][0][2][2], S2_qns_data_K2q[i1][i2][i3][0][3][0], S2_qns_data_K2q[i1][i2][i3][0][3][1], \
			S2_qns_data_K2q[i1][i2][i3][0][3][2], S2_qns_data_K2q[i1][i2][i3][0][3][3], S2_qns_data_K2q[i1][i2][i3][1][1][0], S2_qns_data_K2q[i1][i2][i3][1][2][0], S2_qns_data_K2q[i1][i2][i3][1][2][1], \
			S2_qns_data_K2q[i1][i2][i3][1][3][0], S2_qns_data_K2q[i1][i2][i3][1][3][1], S2_qns_data_K2q[i1][i2][i3][1][3][2]]  = (S_qns_results_K2q[i1][i2][i3].tolist())

def f_opt_K2q_AIO(i1,i2,i3,guess=[0,np.pi,0,np.pi,1,1,1,1]):
	"""
	i1,i2,i3 are index of Omega_list,Tshift_list,g_list respectively 
	return is the optized control theta[1:4] and phi[1:4] that min the  "l2_cost"
	"""
	# min l2_cost
	initial_guess = guess
	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = (['K2q',i1,i2,i3]), method ='Nelder-Mead',options={'maxiter': MAX_ITER})
	return	_opt_.x


print("start opt all K2q:",datetime.now().strftime("%H:%M:%S"))
optimize_K2q_sol=Parallel(n_jobs=n_cores, verbose=0)(delayed(f_opt_K2q_AIO)(i1,i2,i3,(optimize_K2c_sol[i1][i2][i3]).tolist()) for i1 in range(len(Omega_list)) \
	for  i2 in range(len(Tshift_list)) for  i3 in range(len(g_list)))
optimize_K2q_sol = np.array(optimize_K2q_sol).reshape(len(Omega_list),len(Tshift_list),len(g_list), 8) # The last 8=  we have 8 ctrl_params  
np.save('opt_result_K2q.npy',optimize_K2q_sol)
print("end opt all K2q:",datetime.now().strftime("%H:%M:%S"))







############################################################
#    K4c Gaussian classical optimization 
############################################################
from C_qns_K4_L4_classical import * 
obs_avg_simu = np.load('all_meas_results_K4c.npy')
S_qns_results_K4c = np.zeros(np.array(obs_avg_simu).shape, dtype = 'complex_') # create array with same shape
optimize_K4c_sol = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), 8)) # we have 8 ctrl_params 

S1_qns_data_K4c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),          4),dtype = 'complex_')#  
S2_qns_data_K4c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,      4,4),dtype = 'complex_') 
S3_qns_data_K4c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,    4,4,4),dtype = 'complex_') 
S4_qns_data_K4c = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,2,  4,4,4,4),dtype = 'complex_')  

#procedure to have ALL spectra [for all configs, k-order, sign, window etc..]
# notice: the protocols are not in  dimensional form, they are specified in NAMES: Sk_qns_data_Kkc(q)
for i1 in range(len(Omega_list)):
	for i2 in range(len(Tshift_list)):
		for i3 in range(len(g_list)):
			S_qns_results_K4c[i1][i2][i3] = np.ndarray.flatten(O_to_S_matrix_K4c @ (np.matrix(obs_avg_simu[i1][i2][i3]).T) ) # read the obs result for Omega Tshift g
			[S1_qns_data_K4c[i1][i2][i3][0], S1_qns_data_K4c[i1][i2][i3][1], S1_qns_data_K4c[i1][i2][i3][2], S1_qns_data_K4c[i1][i2][i3][3], \
				S2_qns_data_K4c[i1][i2][i3][0][0][0], S2_qns_data_K4c[i1][i2][i3][0][1][0], S2_qns_data_K4c[i1][i2][i3][0][1][1], S2_qns_data_K4c[i1][i2][i3][0][2][0], \
				S2_qns_data_K4c[i1][i2][i3][0][2][1], S2_qns_data_K4c[i1][i2][i3][0][2][2], S2_qns_data_K4c[i1][i2][i3][0][3][0], S2_qns_data_K4c[i1][i2][i3][0][3][1],\
				S2_qns_data_K4c[i1][i2][i3][0][3][2], S2_qns_data_K4c[i1][i2][i3][0][3][3], S3_qns_data_K4c[i1][i2][i3][0][0][1][0][0], \
				S3_qns_data_K4c[i1][i2][i3][0][0][1][1][0], S3_qns_data_K4c[i1][i2][i3][0][0][2][0][0], S3_qns_data_K4c[i1][i2][i3][0][0][2][1][0],\
				 S3_qns_data_K4c[i1][i2][i3][0][0][2][1][1], S3_qns_data_K4c[i1][i2][i3][0][0][2][2][0], S3_qns_data_K4c[i1][i2][i3][0][0][2][2][1],\
				S3_qns_data_K4c[i1][i2][i3][0][0][3][0][0], S3_qns_data_K4c[i1][i2][i3][0][0][3][1][0], S3_qns_data_K4c[i1][i2][i3][0][0][3][1][1], \
				S3_qns_data_K4c[i1][i2][i3][0][0][3][2][0], S3_qns_data_K4c[i1][i2][i3][0][0][3][2][1], S3_qns_data_K4c[i1][i2][i3][0][0][3][2][2], \
				S3_qns_data_K4c[i1][i2][i3][0][0][3][3][0], S3_qns_data_K4c[i1][i2][i3][0][0][3][3][1], S3_qns_data_K4c[i1][i2][i3][0][0][3][3][2], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][1][1][0][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][2][1][0][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][2][1][1][0], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][2][2][0][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][2][2][1][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][2][2][1][1], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][3][1][0][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][1][1][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][2][0][0], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][3][2][1][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][2][1][1], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][2][2][0], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][3][2][2][1], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][0][0], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][1][0], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][1][1], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][2][0], \
				S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][2][1], S4_qns_data_K4c[i1][i2][i3][0][0][0][3][3][2][2]]  = (S_qns_results_K4c[i1][i2][i3].tolist())

def f_opt_K4c_AIO(i1,i2,i3,guess=[0,np.pi,0,np.pi,1,1,1,1]):
	"""
	i1,i2,i3 are index of Omega_list,Tshift_list,g_list respectively 
	return is the optized control theta[1:4] and phi[1:4] that min the  "l2_cost"
	"""	
	# min l2_cost
	initial_guess = guess
	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = (['K4c',i1,i2,i3]), method ='Nelder-Mead',options={'maxiter': MAX_ITER})
	return _opt_.x


print("start opt all K4c:",datetime.now().strftime("%H:%M:%S"))
optimize_K4c_sol=Parallel(n_jobs=n_cores, verbose=0)(delayed(f_opt_K4c_AIO)(i1,i2,i3,(optimize_K2c_sol[i1][i2][i3]).tolist()) for i1 in range(len(Omega_list)) \
	for  i2 in range(len(Tshift_list)) for  i3 in range(len(g_list)))
optimize_K4c_sol = np.array(optimize_K4c_sol).reshape(len(Omega_list),len(Tshift_list),len(g_list), 8) # The last 8=  we have 8 ctrl_params  
np.save('opt_result_K4c.npy',optimize_K4c_sol)
print("end opt all K4c:",datetime.now().strftime("%H:%M:%S"))














############################################################
#    K4q non-G quantum optimization 
############################################################
from C_qns_K4_L4_quantum import * 
obs_avg_simu = np.load('all_meas_results_K4q.npy')
S_qns_results_K4q = np.zeros(np.array(obs_avg_simu).shape, dtype = 'complex_') # create array with same shape
optimize_K4q_sol = np.zeros((len(Omega_list),len(Tshift_list),len(g_list), 8)) # we have 8 ctrl_params 

S1_qns_data_K4q  = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),          4),dtype = 'complex_')# 
S2_qns_data_K4q  = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,      4,4),dtype = 'complex_')
S3_qns_data_K4q  = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,    4,4,4),dtype = 'complex_')
S4_qns_data_K4q  = np.zeros((len(Omega_list),len(Tshift_list),len(g_list),  2,2,2,  4,4,4,4),dtype = 'complex_') 

#procedure to have ALL spectra [for all configs, k-order, sign, window etc..]
# notice: the protocols are not in  dimensional form, they are specified in NAMES: Sk_qns_data_Kkc(q)
for i1 in range(len(Omega_list)):
	for i2 in range(len(Tshift_list)):
		for i3 in range(len(g_list)):
			S_qns_results_K4q [i1][i2][i3] = np.ndarray.flatten(O_to_S_matrix_K4q @ (np.matrix(obs_avg_simu[i1][i2][i3]).T) ) # read the obs result for Omega Tshift g
			[S1_qns_data_K4q[i1][i2][i3][0], S1_qns_data_K4q[i1][i2][i3][1], S1_qns_data_K4q[i1][i2][i3][2], S1_qns_data_K4q[i1][i2][i3][3], \
				S2_qns_data_K4q[i1][i2][i3][0][0][0], S2_qns_data_K4q[i1][i2][i3][0][1][0], S2_qns_data_K4q[i1][i2][i3][0][1][1], \
				S2_qns_data_K4q[i1][i2][i3][0][2][0], S2_qns_data_K4q[i1][i2][i3][0][2][1], S2_qns_data_K4q[i1][i2][i3][0][2][2], S2_qns_data_K4q[i1][i2][i3][0][3][0], S2_qns_data_K4q[i1][i2][i3][0][3][1], \
				S2_qns_data_K4q[i1][i2][i3][0][3][2], S2_qns_data_K4q[i1][i2][i3][0][3][3], S2_qns_data_K4q[i1][i2][i3][1][1][0], S2_qns_data_K4q[i1][i2][i3][1][2][0], S2_qns_data_K4q[i1][i2][i3][1][2][1], \
				S2_qns_data_K4q[i1][i2][i3][1][3][0], S2_qns_data_K4q[i1][i2][i3][1][3][1], S2_qns_data_K4q[i1][i2][i3][1][3][2], S3_qns_data_K4q[i1][i2][i3][0][0][1][0][0], \
				S3_qns_data_K4q[i1][i2][i3][0][0][1][1][0], S3_qns_data_K4q[i1][i2][i3][0][0][2][0][0], S3_qns_data_K4q[i1][i2][i3][0][0][2][1][0], \
				S3_qns_data_K4q[i1][i2][i3][0][0][2][1][1], S3_qns_data_K4q[i1][i2][i3][0][0][2][2][0], S3_qns_data_K4q[i1][i2][i3][0][0][2][2][1],\
				S3_qns_data_K4q[i1][i2][i3][0][0][3][0][0], S3_qns_data_K4q[i1][i2][i3][0][0][3][1][0], S3_qns_data_K4q[i1][i2][i3][0][0][3][1][1], \
				S3_qns_data_K4q[i1][i2][i3][0][0][3][2][0], S3_qns_data_K4q[i1][i2][i3][0][0][3][2][1], S3_qns_data_K4q[i1][i2][i3][0][0][3][2][2], \
				S3_qns_data_K4q[i1][i2][i3][0][0][3][3][0], S3_qns_data_K4q[i1][i2][i3][0][0][3][3][1], S3_qns_data_K4q[i1][i2][i3][0][0][3][3][2], \
				S3_qns_data_K4q[i1][i2][i3][0][1][1][1][0], S3_qns_data_K4q[i1][i2][i3][0][1][2][1][0], S3_qns_data_K4q[i1][i2][i3][0][1][2][2][0], \
				S3_qns_data_K4q[i1][i2][i3][0][1][2][2][1], S3_qns_data_K4q[i1][i2][i3][0][1][3][1][0], S3_qns_data_K4q[i1][i2][i3][0][1][3][2][0], \
				S3_qns_data_K4q[i1][i2][i3][0][1][3][2][1], S3_qns_data_K4q[i1][i2][i3][0][1][3][3][0], S3_qns_data_K4q[i1][i2][i3][0][1][3][3][1], \
				S3_qns_data_K4q[i1][i2][i3][0][1][3][3][2], S3_qns_data_K4q[i1][i2][i3][1][1][2][1][0], S3_qns_data_K4q[i1][i2][i3][1][1][3][1][0], \
				S3_qns_data_K4q[i1][i2][i3][1][1][3][2][0], S3_qns_data_K4q[i1][i2][i3][1][1][3][2][1], S4_qns_data_K4q[i1][i2][i3][0][0][0][1][1][0][0], \
				S4_qns_data_K4q[i1][i2][i3][0][0][0][2][1][0][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][2][1][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][2][2][0][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][2][2][1][0], \
				S4_qns_data_K4q[i1][i2][i3][0][0][0][2][2][1][1], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][1][0][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][1][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][2][0][0], \
				S4_qns_data_K4q[i1][i2][i3][0][0][0][3][2][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][2][1][1], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][2][2][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][2][2][1], \
				S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][0][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][1][1], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][2][0], \
				S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][2][1], S4_qns_data_K4q[i1][i2][i3][0][0][0][3][3][2][2], S4_qns_data_K4q[i1][i2][i3][0][0][1][2][1][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][2][2][1][0], \
				S4_qns_data_K4q[i1][i2][i3][0][0][1][3][1][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][3][2][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][3][2][2][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][3][2][2][1], \
				S4_qns_data_K4q[i1][i2][i3][0][0][1][3][3][1][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][3][3][2][0], S4_qns_data_K4q[i1][i2][i3][0][0][1][3][3][2][1], S4_qns_data_K4q[i1][i2][i3][0][1][1][2][2][1][0], \
				S4_qns_data_K4q[i1][i2][i3][0][1][1][3][2][1][0], S4_qns_data_K4q[i1][i2][i3][0][1][1][3][3][1][0], S4_qns_data_K4q[i1][i2][i3][0][1][1][3][3][2][0], S4_qns_data_K4q[i1][i2][i3][0][1][1][3][3][2][1], \
				S4_qns_data_K4q[i1][i2][i3][1][1][0][3][2][1][0], S4_qns_data_K4q[i1][i2][i3][1][1][1][3][2][1][0]]  = (S_qns_results_K4q[i1][i2][i3].tolist())

def f_opt_K4q_AIO(i1,i2,i3, guess=[0,np.pi,0,np.pi,1,1,1,1]):
	"""
	i1,i2,i3 are index of Omega_list,Tshift_list,g_list respectively 
	return is the optized control theta[1:4] and phi[1:4] that min the  "l2_cost"
	""" 
	# min l2_cost
	initial_guess = guess
	_opt_ = opt.minimize(fun = infidelity,x0= initial_guess,args = (['K4q',i1,i2,i3]), method ='Nelder-Mead',options={'maxiter': MAX_ITER})
	return  _opt_.x


print("start opt all K4q:",datetime.now().strftime("%H:%M:%S"))
optimize_K4q_sol=Parallel(n_jobs=n_cores, verbose=0)(delayed(f_opt_K4q_AIO)(i1,i2,i3,(optimize_K4c_sol[i1][i2][i3]).tolist()) for i1 in range(len(Omega_list)) \
	for  i2 in range(len(Tshift_list)) for  i3 in range(len(g_list)))
optimize_K4q_sol = np.array(optimize_K4q_sol).reshape(len(Omega_list),len(Tshift_list),len(g_list), 8) # The last 8=  we have 8 ctrl_params  
np.save('opt_result_K4q.npy',optimize_K4q_sol)
print("end opt all K4q:",datetime.now().strftime("%H:%M:%S"))



############################################################
#    protocol fidelity in K4q  
############################################################

cpmg_param= [0,1, np.pi, 1, 0, 1, np.pi,1]

fidelity_cpmg = 1-np.array([[[infidelity(cpmg_param, ['K4q',i1,i2,i3] ) for i3 in range(len(g_list)) ] for  i2 in range(len(Tshift_list))] for  i1 in range(len(Omega_list))] )

fidelity_K2c = 1-np.array([[[infidelity(optimize_K2c_sol[i1][i2][i3].tolist(), ['K4q',i1,i2,i3] ) for i3 in range(len(g_list)) ] for  i2 in range(len(Tshift_list))] for  i1 in range(len(Omega_list))] )

fidelity_K2q = 1-np.array([[[infidelity(optimize_K2q_sol[i1][i2][i3].tolist(), ['K4q',i1,i2,i3] ) for i3 in range(len(g_list)) ] for  i2 in range(len(Tshift_list))] for  i1 in range(len(Omega_list))] )

fidelity_K4c = 1-np.array([[[infidelity(optimize_K4c_sol[i1][i2][i3].tolist(), ['K4q',i1,i2,i3] ) for i3 in range(len(g_list)) ] for  i2 in range(len(Tshift_list))] for  i1 in range(len(Omega_list))] )

fidelity_K4q = 1-np.array([[[infidelity(optimize_K4q_sol[i1][i2][i3].tolist(), ['K4q',i1,i2,i3] ) for i3 in range(len(g_list)) ] for  i2 in range(len(Tshift_list))] for  i1 in range(len(Omega_list))] )
