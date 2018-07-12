import numpy as np
from scipy.stats import gamma, norm, multivariate_normal
from scipy.stats import beta as beta_rv
import sys
sys.path.append("../../lib_c/")
from smc2 import smc_c
import matplotlib.pyplot as plt
import time
from numpy.random import multivariate_normal as multi_norm
from scipy.misc import logsumexp
sys.path.append("../useful_functions/")
import useful_functions as uf

def smc2(actions, rewards, idx_blocks, choices, subj_idx, show_progress, apply_rep, apply_weber, beta_softmax, temperature, observational_noise):
	
	assert(2 not in actions); assert(0 in actions); assert(1 in actions); assert(apply_rep==0 or apply_rep==1); assert(apply_weber==0 or apply_weber==1)

	# Extract parameters from task description
	actions    = np.asarray(actions, dtype=np.intc)
	rewards    = np.ascontiguousarray(rewards)
	idx_blocks = np.asarray(idx_blocks, dtype=np.intc)
	N_samples  = 1000
	n_theta    = 1000
	coefficient = .5
	T           = actions.shape[0]
	prev_action   = -1
	upp_bound_eta = 10.

	if apply_rep:
		n_param = 4
	else:
		n_param = 3
	if apply_weber == 1:
		upp_bound_eps = 1.
	else:
		upp_bound_eps = .5

	# samples
	samples                = np.random.rand(n_theta, n_param) 
	if beta_softmax > 0:
		samples[:,1]       = beta_softmax
		sample_beta        = False
		upp_bound_beta     = beta_softmax
		temperature        = False
	else:
		sample_beta        = True
		if temperature:
			upp_bound_beta = np.sqrt(6)/(np.pi * 5)
		else:
			upp_bound_beta = 2.
		samples[:,1]       = np.random.rand(n_theta) * upp_bound_beta
	samples[:,2]           = np.random.rand(n_theta) * upp_bound_eps
	if apply_rep:
		samples[:, 3]      = (2 * np.random.rand(n_theta) - 1) * upp_bound_eta

	# variable memory
	noisy_descendants = np.zeros([n_theta, N_samples, 2])
	noisy_ancestors   = np.zeros([n_theta, N_samples, 2])
	weights_norm      = np.zeros([n_theta, N_samples])
	log_weights_a     = np.zeros([n_theta])
	ancestorsIndexes  = np.ascontiguousarray(np.zeros(n_theta, dtype=np.intc))
	logThetaWeights   = np.zeros(n_theta)
	logThetalkd       = np.zeros(n_theta)
	log_lkd           = np.zeros(n_theta)
	essList           = np.zeros(T)
	acceptance_list   = []
	marg_loglkd       = 0

	#move step variables
	ancestors_indexes_p = np.ascontiguousarray(np.zeros(N_samples, dtype=np.intc))
	samples_new         = np.zeros([n_theta, n_param])
	weights_new         = np.zeros([n_theta, N_samples])
	states_new          = np.zeros([n_theta, N_samples, 2])
	logThetalkd_new     = np.zeros(n_theta)
	state_candidates    = np.zeros([N_samples, 2])
	state_candidates_a  = np.zeros([N_samples, 2])
	weights_candidates  = np.zeros(N_samples)

	# history of samples
	noisy_history      = np.zeros([T,2])

	if show_progress : plt.figure(figsize=(15,9)); plt.suptitle("noisy rl", fontsize=14); plt.ion()
	
	for t_idx in range(T):

		# Print progress
		if (t_idx+1) % 10 == 0 : sys.stdout.write(' ' + str(t_idx+1)); sys.stdout.flush(); print ' marg_loglkd ' + str(marg_loglkd); 

		prev_rew         = np.ascontiguousarray(rewards[:,max(0, t_idx - 1)])
		log_weights_a[:] = logThetaWeights
		if t_idx > 0 and choices[t_idx - 1]:
			assert(actions[max(0, t_idx-1)] == prev_action)

		smc_c.smc_update_2q_c(log_lkd, logThetalkd, noisy_descendants, noisy_ancestors, weights_norm, logThetaWeights, ancestorsIndexes, samples, \
															idx_blocks, choices, prev_action, actions, prev_rew, t_idx, apply_rep, apply_weber, 1, temperature, observational_noise)

		# save and update 
		marg_loglkd += logsumexp(log_weights_a + log_lkd) - logsumexp(log_weights_a)
		normalisedThetaWeights   = uf.to_normalized_weights(logThetaWeights)
		noisy_history[t_idx]     = np.sum((normalisedThetaWeights * np.sum(np.transpose(weights_norm * noisy_descendants.T), axis=1).T), axis=1)

		# Degeneray criterion
		logEss     = 2 * uf.log_sum(logThetaWeights) - uf.log_sum(2 * logThetaWeights)
		essList[t_idx] = np.exp(logEss)

		# update repetition action
		if choices[t_idx] == 1:
			prev_action = actions[t_idx]

		# Move step
		if (essList[t_idx] < coefficient * n_theta):
			acceptance_proba = 0
			if not sample_beta:
				samples_tmp = np.delete(samples, 1, axis=1)
				mu_p        = np.sum(samples_tmp.T * normalisedThetaWeights, axis=1)
				Sigma_p     = np.dot((samples_tmp - mu_p).T * normalisedThetaWeights, (samples_tmp - mu_p))
			else:
				mu_p        = np.sum(samples.T * normalisedThetaWeights, axis=1)
				Sigma_p     = np.dot((samples - mu_p).T * normalisedThetaWeights, (samples - mu_p))

			ancestorsIndexes[:] = uf.stratified_resampling(normalisedThetaWeights)

			for theta_idx in range(n_theta):
				idx_traj = ancestorsIndexes[theta_idx]
				while True:
					sample_cand     = np.array(samples[idx_traj])
					sample_p        = multi_norm(mu_p, Sigma_p)
					sample_p_copy   = np.array(sample_p)
					if (not sample_beta) and apply_rep:
						sample_p    = np.array([sample_p[0], beta_softmax, sample_p[1], sample_p[2]])
						sample_cand = np.delete(sample_cand, 1)
					elif not sample_beta:
						sample_p    = np.array([sample_p[0], beta_softmax, sample_p[1]])
						sample_cand = np.delete(sample_cand, 1)

					if apply_rep:
						if sample_p[0] > 0 and sample_p[0] < 1 and sample_p[1] > 0 and sample_p[1] <= upp_bound_beta \
														and sample_p[2] > 0 and sample_p[2] < upp_bound_eps and sample_p[3] > -upp_bound_eta and sample_p[3] < upp_bound_eta:
							break
					else:
						if sample_p[0] > 0 and sample_p[0] < 1 and sample_p[1] > 0 and sample_p[1] <= upp_bound_beta and sample_p[2] > 0 and sample_p[2] < upp_bound_eps:
							break

				# Launch SMC
				logmarglkd_p = smc_c.smc_2q_c(state_candidates, state_candidates_a, weights_candidates, sample_p, ancestors_indexes_p, \
																		idx_blocks, actions, rewards, choices, t_idx + 1, apply_rep, apply_weber, 1, temperature, observational_noise)

				logAlpha     = np.minimum(0, logmarglkd_p - logThetalkd[idx_traj]  \
													+ get_logtruncnorm(sample_cand, mu_p, Sigma_p) - get_logtruncnorm(sample_p_copy, mu_p, Sigma_p) )

				# accept or reject
				if np.log(np.random.rand()) < logAlpha:
					acceptance_proba          += 1.
					samples_new[theta_idx]     = sample_p
					weights_new[theta_idx]     = weights_candidates
					states_new[theta_idx]      = state_candidates
					logThetalkd_new[theta_idx] = logmarglkd_p
				else:
					samples_new[theta_idx]     = samples[idx_traj]
					weights_new[theta_idx]     = weights_norm[idx_traj]
					states_new[theta_idx]      = noisy_descendants[idx_traj]
					logThetalkd_new[theta_idx] = logThetalkd[idx_traj]

			print ('\n')
			print ('acceptance ratio is ')
			print (acceptance_proba/n_theta)
			print ('\n')
			acceptance_list.append(acceptance_proba/n_theta)

			weights_norm[:]               = weights_new
			logThetalkd[:]                = logThetalkd_new
			logThetaWeights[:]            = np.zeros(n_theta)
			noisy_descendants[:]          = states_new
			samples[:]                    = samples_new
			normalisedThetaWeights        = uf.to_normalized_weights(logThetaWeights)


		if show_progress and t_idx % 10:
			plt.subplot(3,2,1)
			plt.plot(range(t_idx), noisy_history[:t_idx,0], 'r'); plt.hold(True)
			plt.plot(range(t_idx), noisy_history[:t_idx,1], 'b'); 
			plt.hold(False)
			plt.xlabel('trials')
			plt.ylabel('Q-value 0 (red), and 1 (blue)')

			plt.subplot(3,2,4)
			plt.plot(range(t_idx), essList[:t_idx], 'b', linewidth=2); plt.hold(True)
			plt.plot(plt.gca().get_xlim(), [n_theta/2,  n_theta/2],'b--', linewidth=2)
			plt.axis([0, t_idx-1, 0, n_theta])
			plt.hold(False)
			plt.xlabel('trials')
			plt.ylabel('ess')


			if temperature:
				mean_beta = np.sum(normalisedThetaWeights * (1./samples[:,1]))
				std_beta  = np.sqrt(np.sum(normalisedThetaWeights * (1./samples[:,1])**2) - mean_beta**2)				
				x         = np.linspace(0.,200,5000)
			else:
				mean_beta = np.sum(normalisedThetaWeights * (10**samples[:,1]))
				std_beta  = np.sqrt(np.sum(normalisedThetaWeights * (10**samples[:,1])**2) - mean_beta**2)
				x = np.linspace(0.,10**upp_bound_beta,5000)
			plt.subplot(3,2,3)
			plt.plot(x, norm.pdf(x, mean_beta, std_beta), 'g'); plt.hold(True)
			plt.plot([mean_beta, mean_beta], plt.gca().get_ylim(),'g', linewidth=2)
			plt.hold(False)
			plt.xlabel('beta softmax')
			plt.ylabel('pdf')

			mean_alpha_0 = np.sum(normalisedThetaWeights * samples[:,0])
			std_alpha_0  = np.sqrt(np.sum(normalisedThetaWeights * samples[:,0]**2) - mean_alpha_0**2)
			mean_alpha_1 = np.sum(normalisedThetaWeights * samples[:,0])
			std_alpha_1  = np.sqrt(np.sum(normalisedThetaWeights * samples[:,0]**2) - mean_alpha_1**2)
			plt.subplot(3,2,2)
			x = np.linspace(0.,1.,5000)
			plt.plot(x, norm.pdf(x, mean_alpha_0, std_alpha_0), 'm'); plt.hold(True)
			plt.plot([mean_alpha_0, mean_alpha_0], plt.gca().get_ylim(), 'm')
			plt.plot(x, norm.pdf(x, mean_alpha_1, std_alpha_1), 'c')
			plt.plot([mean_alpha_1, mean_alpha_1], plt.gca().get_ylim(), 'c')
			plt.hold(False)
			plt.xlabel('learning rates')
			plt.ylabel('pdf')

			mean_epsilon = np.sum(normalisedThetaWeights * samples[:,2])
			std_epsilon  = np.sqrt(np.sum(normalisedThetaWeights * samples[:,2]**2) - mean_epsilon**2)
			plt.subplot(3, 2, 6);
			x = np.linspace(0.,upp_bound_eps,5000)
			if apply_rep == 1:
				mean_rep = np.sum(normalisedThetaWeights * samples[:,3])
				std_rep  = np.sqrt(np.sum(normalisedThetaWeights * samples[:,3]**2) - mean_rep**2)
				x        = np.linspace(-2.,2.,5000)
				plt.plot(x, norm.pdf(x, mean_rep, std_rep), 'y'); plt.hold(True)
				plt.plot([mean_rep, mean_rep], plt.gca().get_ylim(),'y', linewidth=2)
			plt.plot(x, norm.pdf(x, mean_epsilon, std_epsilon), 'g'); plt.hold(True)
			plt.plot([mean_epsilon, mean_epsilon], plt.gca().get_ylim(), 'g', linewidth=2)
			plt.hold(False)
			plt.xlabel('epsilon std (green), rep_bias (yellow)')
			plt.ylabel('pdf')
			plt.draw()
			plt.show()
			plt.pause(0.05)
		
	return [samples, noisy_history, acceptance_list, normalisedThetaWeights, logThetalkd,  marg_loglkd]


def get_logtruncnorm(sample, mu, sigma):
	return multivariate_normal.logpdf(sample, mu, sigma)


















































