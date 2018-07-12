import numpy as np
from scipy.stats import gamma, norm
import sys
from state_estimation import smc_c
import matplotlib.pyplot as plt
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf

def get_smoothing_trajectory(actions, rewards, idx_blocks, choices, subj_idx, m_sample, apply_rep_bias, apply_weber, apply_guided):

	N_samples  = 10000
	M_samples  = 2000
	T          = actions.shape[0]
	n_essay    = actions.shape[0]
	assert(apply_rep_bias == 0)
	temperature = True

	# params
	alpha_c, alpha_u, beta, epsilon = m_sample[0], m_sample[1], m_sample[2], m_sample[3]
		
	# variable memory
	traj_noisy          = np.zeros([N_samples, n_essay, 2]);
	weights_norm        = np.zeros(N_samples)
	weights_unnorm      = np.zeros([N_samples, n_essay])
	ancestors_indexes_p = np.ascontiguousarray(np.zeros(N_samples, dtype=np.intc))

	# save variables
	backward_samples    = np.zeros([M_samples, n_essay, 2])

	# conditional smc
	margloglkd = smc_c.smc_2q_c(traj_noisy, weights_unnorm, weights_norm, alpha_c, alpha_u, beta, epsilon, ancestors_indexes_p, idx_blocks, actions, rewards, choices, T, apply_rep_bias, apply_weber, apply_guided)
	
	# backward sampling
	smc_c.backward_smc_2q_c(traj_noisy, backward_samples, weights_norm, alpha_c, alpha_u, epsilon, weights_unnorm, idx_blocks, rewards, choices, apply_weber)

	return [backward_samples, margloglkd]

def get_noiseless(actions, rewards, idx_blocks, noisy, alpha_c, alpha_u):
	nb_samples, T = noisy.shape[:-1]
	noiseless  = np.zeros([nb_samples, T, 2])
	for t_idx in range(T):
		if idx_blocks[t_idx] == 0.:
			prev_rew = rewards[:,t_idx - 1]
			if actions[t_idx - 1] == 0:
				noiseless[:, t_idx, 0] = (1 - alpha_c) * noisy[:, t_idx - 1, 0] + alpha_c * prev_rew[0]
				noiseless[:, t_idx, 1] = (1 - alpha_u) * noisy[:, t_idx - 1, 1] + alpha_u * prev_rew[1]
			else:
				noiseless[:, t_idx, 0] = (1 - alpha_u) * noisy[:, t_idx - 1, 0] + alpha_u * prev_rew[0]
				noiseless[:, t_idx, 1] = (1 - alpha_c) * noisy[:, t_idx - 1, 1] + alpha_c * prev_rew[1]
		else:
			noiseless[:, t_idx, :] = 0.5
	return noiseless


def get_rl_traj(actions, rewards, idx_blocks, alpha_c, alpha_u):
	T          = len(idx_blocks)
	noiseless  = np.zeros([T, 2])
	for t_idx in range(T):
		if idx_blocks[t_idx] == 0.:
			prev_rew = rewards[:,t_idx - 1]
			if actions[t_idx - 1] == 0:
				noiseless[t_idx, 0] = (1 - alpha_c) * noiseless[t_idx - 1, 0] + alpha_c * prev_rew[0]
				noiseless[t_idx, 1] = (1 - alpha_u) * noiseless[t_idx - 1, 1] + alpha_u * prev_rew[1]
			else:
				noiseless[t_idx, 0] = (1 - alpha_u) * noiseless[t_idx - 1, 0] + alpha_u * prev_rew[0]
				noiseless[t_idx, 1] = (1 - alpha_c) * noiseless[t_idx - 1, 1] + alpha_c * prev_rew[1]
		else:
			noiseless[t_idx, :] = 0.5
	return noiseless
