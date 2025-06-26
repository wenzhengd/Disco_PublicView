import numpy as np
#from qutip import *
#from joblib import Parallel, delayed
#from datetime import datetime
from operator import add
import sys
#from scipy import interpolate
#from scipy.integrate import quad, dblquad,nquad
#from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
#os.chdir('/Users/wenzheng/Dropbox (Dartmouth College)/Code/check quantum noise')
#print("Current working directory: {0}".format(os.getcwd()))

#from CA_params_setting import *




def _2qb_S2_plot(data_qns, data_theory , label_list, fontsize_list):
	"""
	This will produce a (2,4) plot gird, 1st row S_qns, 2nd row S_theory; 
	the colms are q: AA, AB, BA, BB respectively
	--------------------------------------------
	data_XX_: an 2d-array of spectra, with a unique g specified
	label_list: a list of strings that goes to title or label
	"""
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30, 12))
	ticks = ['1', '2', '3', '4']
	font_small, font_mid, font_large = fontsize_list[0], fontsize_list[1], fontsize_list[2]
	for row in range(2): # one for S_qns one for S_theory
		for col in range(4): # AA AB, BA, BB
			[q1,q2] = f'{col:02b}'
			_S_qns_ = np.real (data_qns[int(q1)][int(q2)]) # data will be plotted top
			_S_theo_ = np.real(data_theory)				 # data will be plotted bottom
			v_min = min(np.min(_S_qns_), np.min(np.real(_S_theo_)))
			v_max = max(np.max(_S_qns_), np.max(np.real(_S_theo_)))
			im = axes[row][col].matshow(np.ma.masked_equal(_S_qns_,0), vmin=v_min, vmax= v_max) if row==0 \
					else axes[row][col].matshow(np.ma.masked_equal(np.real(_S_theo_),0), vmin=v_min, vmax= v_max)
			axes[row][col].set_title("K8c "+r' $ \hat{ \bar{S}}^{(0)}_{q1q2}(n_1,n_2)_{\text{QNS}} \quad $' + label_list[0],fontsize = font_small) if row ==0 else\
				axes[row][col].set_title("K8c "+r' $  \bar{S}^{(0)}_{q1q2}(n_1,n_2)_{\text{true}} \quad $' +  label_list[0],fontsize = font_small)
			axes[row][col].set_title("K8c "+r' $ \bar{S}^{(0)}_{AA}(n_1,n_2) \quad$' +  label_list[0],fontsize = font_small) if [q1,q2]==['0','0'] else \
			    axes[row][col].set_title("K8c "+r' $ \bar{S}^{(0)}_{AB}(n_1,n_2) \quad $'+ label_list[0],fontsize = font_small) if [q1,q2]==['0','1'] else \
			    axes[row][col].set_title("K8c "+r' $ \bar{S}^{(0)}_{BA}(n_1,n_2) \quad $'+  label_list[0],fontsize = font_small) if [q1,q2]==['1','0'] else\
			    axes[row][col].set_title("K8c "+r' $ \bar{S}^{(0)}_{BB}(n_1,n_2) \quad $'+  label_list[0],fontsize = font_small)
			axes[row][col].set_xlabel(r'$n_2$',size=font_mid)
			axes[row][col].set_ylabel(r'$n_1$',size=font_mid)
			axes[row][col].set_xticklabels(['']+ticks,size=font_mid)
			axes[row][col].set_yticklabels(['']+ticks,size=font_mid)
			axes[row][col].tick_params(axis='both', which='major', labelsize=font_mid)
			cb = fig.colorbar(im,ax=axes[row][col])
			cb.ax.tick_params(labelsize=font_small)



from matplotlib import gridspec
def _2qb_S4_plot(data_qns, data_theory , label_list, fontsize_list):
	"""
	This will produce a  l & righg two panels plot, left panel S_qns, right one S_theory; 
	each panel consists of  (4,4) =16 blocks, representing 16 q values [labeled onsite].
	Each block represents a locality well-defined S4 spectra; which can furtuer be (2,2) zoned as (n1,n2) = [[(1,1), (1,2)],[(2,1),(2,2)]]
	--------------------------------------------
	data_XX_: an 4d-array of spectra, with a unique g specified
	label_list: a list of strings that goes to title or label
	"""

	fig = plt.figure(constrained_layout=True,figsize=(16, 8))
	gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1 )
	
	gsl = gridspec.GridSpecFromSubplotSpec(4, 4, gs0[0],hspace=0.0,wspace=0.0)
	gsr = gridspec.GridSpecFromSubplotSpec(4, 4, gs0[1],hspace=0.0,wspace=0.0)

	font_small, font_mid, font_large = fontsize_list[0], fontsize_list[1], fontsize_list[2]
	
	plt.suptitle("<<<< === QNS   " +r' $ S^{(000)}(n_1,n_2,n_3,n_4)$' + label_list[0] + "    theory    >>>>", fontsize=font_large)
	
	v_max = max(np.max(data_qns), np.max(data_theory)) #max(np.max(S4_ppp_qns_results),np.max(s4_ppp_theory))
	v_min = min(np.min(data_qns), np.min(data_theory))
	ticks = ['1', '2', '1', '2']
	
	axs = []
	for i in range(16):
		ax = fig.add_subplot(gsl[i])
		[q1,q2,q3,q4] = f'{i:04b}'
		_S_qns_ =data_qns[int(q1)][int(q2)][int(q3)][int(q4)]
		_S_qns_ =np.reshape(np.ndarray.flatten(np.array([[_S_qns_[0][0],_S_qns_[0][1]],[_S_qns_[1][0],_S_qns_[1][1]]]) ), (4,4))
		im = ax.matshow(np.ma.masked_equal(_S_qns_, 0), vmin=v_min, vmax= v_max)
		if (i-np.mod(i,4))//4 ==3:
			ax.set_xlabel(r'$n_4$',size=font_mid)
		if np.mod(i,4) ==0:   
			ax.set_ylabel(r'$n_3$',size=font_mid)        
		ax.set_xticklabels(['']+ticks if  (i-np.mod(i,4))//4+1==1 else [''],size=font_mid)  # only x_tick the top row
		ax.set_yticklabels(['']+ticks if np.mod(i,4)+1==1 else [''],size=font_mid) # only y_tick the left col
		#ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
		f = lambda q: 'A' if q=='0' else 'B' # map the locality index  to A or B
		ax.set_title(f(q1)+f(q2)+f(q3)+f(q4), fontsize = font_mid, x=0.65, y=0.6 )
		ax.tick_params(axis='both', which='major', labelsize=font_mid)
		axs += [ax]
	    
	for i in range(16):
		ax = fig.add_subplot(gsr[i])
		[q1,q2,q3,q4] = f'{i:04b}'
		_S_theo_ = data_theory
		_S_theo_ =np.reshape(np.ndarray.flatten(np.array([[_S_theo_[0][0],_S_theo_[0][1]],[_S_theo_[1][0],_S_theo_[1][1]]]) ), (4,4))
		im = ax.matshow(np.ma.masked_equal(_S_theo_, 0), vmin=v_min, vmax= v_max)
		if (i-np.mod(i,4))//4 ==3:
			ax.set_xlabel(r'$n_4$',size=font_mid)
		if np.mod(i,4) ==0:   
			ax.set_ylabel(r'$n_3$',size=font_mid)        
		ax.set_xticklabels(['']+ticks if  (i-np.mod(i,4))//4+1==1 else [''],size=font_mid)  # only x_tick the top row
		ax.set_yticklabels(['']+ticks if np.mod(i,4)+1==1 else [''],size=font_mid) # only y_tick the left col
		#ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
		f = lambda q: 'A' if q=='0' else 'B' # map the locality index  to A or B
		ax.set_title(f(q1)+f(q2)+f(q3)+f(q4), fontsize = font_mid, x=0.65, y=0.6 )
		ax.tick_params(axis='both', which='major', labelsize=font_mid)
		axs += [ax]    
	
	cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])    
	fig.colorbar(im, cbar_ax, ax=axs )
	
	#plt.show()






