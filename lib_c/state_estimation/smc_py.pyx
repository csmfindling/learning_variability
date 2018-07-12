import cython
import numpy as np
cimport numpy as np

cdef extern from "smc_functions.hpp" namespace "smc":

	void backward_smc_2q(double* traj_noisy, double* bw_noisy, double* weights, double alpha_c, double alpha_n, double epsilon, double* weights_unnorm, int* blocks, double* rewards, int* choices, int apply_weber, int T, int N_samples, int N_backward_samples);

	double smc_2q(double* Q_values, double* weights_unnorm, double* weights_res, double alpha_c, double alpha_u, double beta_softmax, double epsilon, int* ancestors_indexes, int N_samples, int T, int n_essay, int* blocks, int* actions, double* rewards, int* choices, int apply_rep_bias, int apply_weber, int apply_guided);


@cython.boundscheck(False)
@cython.wraparound(False)

def smc_2q_c(np.ndarray[double, ndim=3, mode="c"] Q_values not None, np.ndarray[double, ndim=2, mode="c"] weights_unnorm not None, np.ndarray[double, ndim=1, mode="c"] weights_res not None,\
			 double alpha_c, double alpha_u, double beta_softmax, double epsilon, np.ndarray[int, ndim=1, mode="c"] ancestors_indexes not None, \
			np.ndarray[int, ndim=1, mode="c"] blocks not None, np.ndarray[int, ndim=1, mode="c"] actions not None, np.ndarray[double, ndim=2, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] choices not None, int T, int apply_rep_bias, int apply_weber, int apply_guided):
	cdef int N_samples, n_essay
	N_samples = Q_values.shape[0]
	n_essay   = len(actions)
	return smc_2q(&Q_values[0,0,0], &weights_unnorm[0,0], &weights_res[0], alpha_c, alpha_u, beta_softmax, epsilon, &ancestors_indexes[0], N_samples, T, n_essay, &blocks[0], &actions[0], &rewards[0,0], &choices[0], apply_rep_bias, apply_weber, apply_guided)

def backward_smc_2q_c(np.ndarray[double, ndim=3, mode="c"] traj_noisy not None, np.ndarray[double, ndim=3, mode="c"] bw_noisy not None,  \
						np.ndarray[double, ndim=1, mode="c"] weights not None, double alpha_c, double alpha_n, double epsilon, np.ndarray[double, ndim=2, mode="c"] weights_unnorm not None, \
						np.ndarray[int, ndim=1, mode="c"] blocks not None, np.ndarray[double, ndim=2, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] choices not None, int apply_weber):
	cdef int N_samples, T, M_samples
	N_samples = traj_noisy.shape[0]; T = traj_noisy.shape[1]
	M_samples = bw_noisy.shape[0]

	return backward_smc_2q(&traj_noisy[0,0,0], &bw_noisy[0,0,0], &weights[0], alpha_c, alpha_n, epsilon, &weights_unnorm[0,0], &blocks[0], &rewards[0,0], &choices[0], apply_weber, T, N_samples, M_samples)