import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf
import warnings

# simulate RL model with learning variability
def simulate_noisy_rl(rewards, complete, idx_blocks, choices, forced_actions, sample, apply_rep_bias, apply_weber, apply_observational_noise):
	T                  = len(choices)
	noisy_trajectories = np.zeros([T, 2])
	actions_simul      = np.zeros([T])
	prev_act		   = - 1
	alpha_c            = sample[0]
	alpha_u            = sample[1]
	beta_softmax       = sample[2]
	epsilon            = sample[3]
	actions            = np.zeros(T) - 1
	prev_rew           = np.zeros(2)
	observed_rewards   = np.zeros(rewards.shape) - np.inf
	if apply_rep_bias:
		rep_bias = sample[4]
	for t_idx in range(T):

		if t_idx > 0:
			if complete:
				prev_rew = rewards[:, t_idx - 1]
			else:
				prev_rew[int(actions_simul[t_idx - 1])]     = rewards[int(actions_simul[t_idx - 1]), t_idx - 1]
				prev_rew[1 - int(actions_simul[t_idx - 1])] = .5

			observed_rewards[:, t_idx - 1] = prev_rew	


		if idx_blocks[t_idx] == 0.:

			if actions_simul[t_idx - 1] == 0:
				mu0 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
				mu1 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
			else:
				mu0 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
				mu1 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]

			if apply_weber == 1:
				noise_level0 = np.abs(prev_rew[0] - noisy_trajectories[t_idx - 1, 0]) * epsilon
				noise_level1 = np.abs(prev_rew[1] - noisy_trajectories[t_idx - 1, 1]) * epsilon
			else:
				noise_level0 = epsilon
				noise_level1 = epsilon

			if apply_observational_noise==0 and choices[t_idx - 1]==0:
				noise_level0 = 0
				noise_level1 = 0

			noisy_trajectories[t_idx, 0] = mu0 + noise_level0 * np.random.normal()
			noisy_trajectories[t_idx, 1] = mu1 + noise_level1 * np.random.normal()

			if choices[t_idx] == 1:
				# probability to choose 1
				if (apply_rep_bias==0) or (prev_act==-1):
					proba_1 = 1./(1. + np.exp(beta_softmax * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1])))
				else:
					proba_1 = 1./(1. + np.exp(beta_softmax * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1]) - np.sign(prev_act - .5) * rep_bias))
				
				# simulate action
				if np.random.rand() < proba_1:
					actions_simul[t_idx] = 1
				else:
					actions_simul[t_idx] = 0

				# prev action
				prev_act = actions_simul[t_idx]
			else:
				actions_simul[t_idx] = forced_actions[t_idx]

		else:
			noisy_trajectories[t_idx] = 0.5
			prev_act                  = -1

			if choices[t_idx] == 1:
				if np.random.rand() < .5:
					actions_simul[t_idx] = 1
				else:
					actions_simul[t_idx] = 0

					# previous action
					prev_act = actions_simul[t_idx]
			else:
				actions_simul[t_idx]  = forced_actions[t_idx]

	if t_idx > 0:
		if complete:
			prev_rew = rewards[:, t_idx]
		else:
			prev_rew[int(actions_simul[t_idx])]     = rewards[int(actions_simul[t_idx]), t_idx]
			prev_rew[1 - int(actions_simul[t_idx])] = .5

		observed_rewards[:, t_idx] = prev_rew	

	return noisy_trajectories, actions_simul, observed_rewards

# simulate standard noise-free RL model
def simulate_noiseless_rl(rewards, complete, idx_blocks, choices, forced_actions, sample, apply_rep_bias, apply_weber_decision_noise):
	assert(apply_weber_decision_noise == 0), 'Simulation not developped for apply_weber_decision_noise = 1'

	T                      = len(choices)
	noiseless_trajectories = np.zeros([T, 2])
	actions_simul          = np.zeros([T])
	prev_act	    	   = - 1
	alpha_c                = sample[0]
	alpha_u                = sample[1]
	beta_softmax           = sample[2]
	prev_rew               = np.zeros(2)
	observed_rewards       = np.zeros(rewards.shape) - np.inf
	if apply_rep_bias:
		rep_bias = sample[3]

	for t_idx in range(T):

		if t_idx > 0:
			if complete:
				prev_rew = rewards[:, t_idx - 1]
			else:
				prev_rew[int(actions_simul[t_idx - 1])]     = rewards[int(actions_simul[t_idx - 1]), t_idx - 1]
				prev_rew[1 - int(actions_simul[t_idx - 1])] = .5

			observed_rewards[:, t_idx - 1] = prev_rew	

		if idx_blocks[t_idx] == 0:
				
			if actions_simul[t_idx - 1] == 0:
				noiseless_trajectories[t_idx, 0] = (1 - alpha_c) * noiseless_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
				noiseless_trajectories[t_idx, 1] = (1 - alpha_u) * noiseless_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
			else:
				noiseless_trajectories[t_idx, 0] = (1 - alpha_u) * noiseless_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
				noiseless_trajectories[t_idx, 1] = (1 - alpha_c) * noiseless_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]

		else:
			prev_act                         = -1
			noiseless_trajectories[t_idx]    = 0.5

		if choices[t_idx] == 1:
			if (apply_rep_bias==0) or (prev_act==-1):
				proba_1 = 1./(1. + np.exp(beta_softmax * (noiseless_trajectories[t_idx, 0] - noiseless_trajectories[t_idx, 1])))
			else:
				proba_1 = 1./(1. + np.exp(beta_softmax * (noiseless_trajectories[t_idx, 0] - noiseless_trajectories[t_idx, 1]) - np.sign(prev_act - .5) * rep_bias))
			if np.random.rand() < proba_1:
				actions_simul[t_idx] = 1
			else:
				actions_simul[t_idx] = 0
			prev_act = actions_simul[t_idx]
		else:
			actions_simul[t_idx] = forced_actions[t_idx]

	if t_idx > 0:
		if complete:
			prev_rew = rewards[:, t_idx]
		else:
			prev_rew[int(actions_simul[t_idx])]     = rewards[int(actions_simul[t_idx]), t_idx]
			prev_rew[1 - int(actions_simul[t_idx])] = .5

		observed_rewards[:, t_idx] = prev_rew	
		
	return noiseless_trajectories, actions_simul, observed_rewards














