import cython
import numpy as np
cimport numpy as np

cdef extern from "smc_functions.hpp" namespace "smc":

	void smc_update_2q(double* log_lkd, double* logThetaLks, double* noisy_descendants, double* noisy_ancestors, int* ancestorsIndexes, double* weights_unnorm, double* logThetaWeights, double* theta_samples, int n_theta, int N_samples, int t_idx, int* blocks, int* choices, int prev_act, int* actions, double* prev_rew, int apply_rep_bias, int apply_weber, int nbAlpha, int temperature, int observational_noise);

	double smc_2q(double* Q_values, double* Q_values_ancestors, double* weights_res, double* sample, int* ancestors_indexes, int N_samples, int T, int n_essay, int* blocks, int* actions, double* rewards, int* choices, int apply_rep_bias, int apply_weber, int nbAlpha, int temperature, int observational_noise);
	
@cython.boundscheck(False)
@cython.wraparound(False)


def smc_update_2q_c(np.ndarray[double, ndim=1, mode="c"] log_lkd not None, np.ndarray[double, ndim=1, mode="c"] logThetaLks not None, np.ndarray[double, ndim=3, mode="c"] noisy_descendants not None, np.ndarray[double, ndim=3, mode="c"] noisy_ancestors not None, \
						np.ndarray[double, ndim=2, mode="c"] weights_unnorm not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, \
						np.ndarray[int, ndim=1, mode="c"] ancestorsIndexes not None, np.ndarray[double, ndim=2, mode="c"] theta_samples not None, \
						np.ndarray[int, ndim=1, mode="c"] blocks not None, np.ndarray[int, ndim=1, mode="c"] choices not None, int prev_act, np.ndarray[int, ndim=1, mode="c"] actions not None, np.ndarray[double, ndim=1, mode="c"] prev_rew not None, int t_idx, int apply_rep_bias, int apply_weber, int nbAlpha, int temperature, int observational_noise):
	cdef int N_samples, n_theta
	n_theta = noisy_descendants.shape[0]; N_samples = noisy_descendants.shape[1]

	return smc_update_2q(&log_lkd[0], &logThetaLks[0], &noisy_descendants[0,0,0], &noisy_ancestors[0,0,0], &ancestorsIndexes[0], &weights_unnorm[0,0], &logThetaWeights[0], &theta_samples[0,0], n_theta, N_samples, t_idx, &blocks[0], &choices[0], prev_act, &actions[0], &prev_rew[0], apply_rep_bias, apply_weber, nbAlpha, temperature, observational_noise);


def smc_2q_c(np.ndarray[double, ndim=2, mode="c"] Q_values not None, np.ndarray[double, ndim=2, mode="c"] Q_values_ancestors not None, \
			np.ndarray[double, ndim=1, mode="c"] weights_res not None, np.ndarray[double, ndim=1, mode="c"] sample not None, np.ndarray[int, ndim=1, mode="c"] ancestors_indexes not None, \
			np.ndarray[int, ndim=1, mode="c"] blocks not None, np.ndarray[int, ndim=1, mode="c"] actions not None, np.ndarray[double, ndim=2, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] choices not None, int T, int apply_rep_bias, int apply_weber, int nbAlpha, int temperature, int observational_noise):
	cdef int N_samples, n_essay
	N_samples = Q_values.shape[0]
	n_essay   = len(actions)
	return smc_2q(&Q_values[0,0], &Q_values_ancestors[0,0], &weights_res[0], &sample[0], &ancestors_indexes[0], N_samples, T, n_essay, &blocks[0], &actions[0], &rewards[0,0], &choices[0], apply_rep_bias, apply_weber, nbAlpha, temperature, observational_noise)
