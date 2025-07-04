# Classical K2 QNS switching function
import numpy as np
"""
see Mathematica file called "D2_sample1000_enough.nb"
"""

y_qns_K2c = np.array([[[1,0,0],[1,0,0],[1,0,0],[1,0,0]],[[1,0,0],[1,0,0],[1,0,0],[1,0,0]],[[1,0,0],[1,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,0,0],\
	[1,0,0],[0,1,0]],[[1,0,0],[1,0,0],[1,0,0],[0,0,1]],[[1,0,0],[1,0,0],[0,1,0],[1,0,0]],[[1,0,0],[1,0,0],[0,1,0],[1,0,0]],[[1,0,0],[1,0,0],[0,1,0],\
	[0,1,0]],[[1,0,0],[1,0,0],[0,0,1],[1,0,0]],[[1,0,0],[0,1,0],[1,0,0],[1,0,0]],[[1,0,0],[0,1,0],[1,0,0],[1,0,0]],[[1,0,0],[0,1,0],[1,0,0],[0,1,0]],\
	[[1,0,0],[0,1,0],[0,1,0],[1,0,0]],[[1,0,0],[0,0,1],[1,0,0],[1,0,0]]])




init_list_K2c = [2,3,2,3,2,2,3,2,2,2,3,2,2,2]


O_to_S_matrix_K2c = np.array([[0., -0.5, 0., 0.25, 0., 0., 0.25, 0., 0., 0., 0.25, 0., 0., 0.], [0., 0.25, 0., 0., 0., 0., 0., 0., 0., 0., -0.25, 0., 0., 0.],\
  [0., 0.25, 0., 0., 0., 0., -0.25, 0., 0., 0., 0., 0., 0.,  0.], [0., 0.25, 0., -0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], \
  [0.125, 0., 0.25, 0., -0.25, 0.25,  0., -0.125, -0.25, 0.25, 0., -0.125, -0.125, -0.25],[0., 0., -0.125, 0., 0., -0.125, 0., 0., 0., 0.,0., 0.125, 0.125, 0.], \
  [0., 0., 0., 0., 0., 0., 0., 0., 0., -0.25, 0., 0., 0., 0.25], [0., 0., -0.125, 0., 0., 0., 0., 0.125,  0., -0.125, 0., 0., 0.125, 0.], \
  [-0.125, 0., 0., 0., 0., 0.125, 0., 0., 0.,  0.125, 0., 0., -0.125, 0.], [0., 0., 0., 0., 0., -0.25, 0., 0., 0.25, 0., 0., 0., 0., 0.], \
 [0., 0., 0., 0., 0., -0.125, 0., 0.125, 0., -0.125, 0., 0.125, 0., 0.], [-0.125, 0., 0.125, 0., 0., 0., 0., 0., 0., 0.125, 0., -0.125, 0., 0.],\
  [-0.125, 0., 0.125, 0., 0., 0.125, 0., -0.125, 0., 0., 0., 0., 0., 0.], [0., 0., -0.25, 0., 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
