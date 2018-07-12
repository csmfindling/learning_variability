import warnings
import numpy as np
import pickle as pkl
import sys
sys.path.append("twoq_value_2alpha/useful_functions/")
sys.path.append("lib_py/twoq_value_2alpha/useful_functions/")
sys.path.append("lib_c/")
sys.path.append("lib_py/lib_c/")
import useful_functions as useful_functions
from scipy.stats import pearsonr
from scipy.io import savemat
from twoq_value_2alpha.simulation import simulation
from twoq_value_2alpha.state_estimation import state_estimation_2q as state_estimation_2q_2alpha
from twoq_value_2alpha.smc2 import smc2_2q as smc2_2q_2alpha
from twoq_value_2alpha.ibis import ibis_2q as ibis_2q_2alpha
from twoq_value_1alpha.state_estimation import state_estimation_2q as state_estimation_2q_1alpha
from twoq_value_1alpha.smc2 import smc2_2q as smc2_2q_1alpha
from twoq_value_1alpha.ibis import ibis_2q as ibis_2q_1alpha

"""
Class smc_object summarizes all of the functions available for inference and smoothing problems.
"""
class smc_object():
	'''
	---- constructor -----	
	Parameters:
	- a dictionary `info' with keys:
		+ actions    : subjects actions (0 or 1)
		+ blocks_idx : array of 0 of same length then the actions with a 1 at the beginning of every new block
		+ choices    : whether the choices was a subject choice (1) or a forced choice (0) ; by default, array of ones us assumed
		+ subj_idx   : subject id. Important for saving purposes
		+ rewards    : rewards of shape [2, nbOfTrials]. These should be normalised between 0 and 1. 
					If partial, the learning rule of the unchosen option will override the unchosen reward
	- complete = 1/0      : whether we are in a complete or partial feedback
	- leaky    = -1/0/1   : when complete, nothing to do, leaky=-1. In partial, one must choose between 
						  leaky model or anticorrelated model leaky = 1/0 in partial feedback : if leaky == 1, 
						  then the regression to the mean model will be applied 
						  elif leaky == 0 then the 1 - R model will the applied else an error will be raise
	- onload = True/False : If onload is set to True, the dictionary input expects much more variables (see load_results function). 
						   Essentially, it expects the contents of the output of the save function
	'''
	def __init__(self, info, complete=1, leaky=-1, onload = False):
		if complete not in [0,1]:
			raise SyntaxError('complete=1 or complete=0')
		if not onload:
			self.leaky          = leaky
			self.complete       = complete
			self.inference_done = False
			self.got_map        = False
			self.got_traj       = False
			self.got_simul      = False
			self.traj_param     = {}
			self.param_names    = []
			if 'actions' in info.keys():
				self.actions = np.asarray(info['actions'], dtype=np.intc)
			else:
				raise ValueError('Expected an actions field in input dictionary')
			if 'blocks_idx' in info.keys():
				self.idx_blocks = np.asarray(info['blocks_idx'], dtype=np.intc)
			else:
				self.idx_blocks = np.zeros(len(self.actions)); self.idx_blocks[0] = 1.
			if 'choices' in info.keys():
				self.choices = np.asarray(info['choices'], dtype = np.intc)
			else:
				self.choices = np.ones(len(self.actions), dtype = np.intc)
			if 'subject_idx' in info.keys():
				self.subj_idx = info['subject_idx'] 
			elif 'subj_idx' in info.keys():
				self.subj_idx = info['subj_idx']
			else:
				self.subj_idx = 0				
			if 'rewards' in info.keys():
				if leaky==1:
					rewards_0         = info['rewards'][0] * (self.actions==0) + .5 * (self.actions==1)
					rewards_1         = info['rewards'][1] * (self.actions==1) + .5 * (self.actions==0)
				elif leaky==0:
					rewards_0         = info['rewards'][0] * (self.actions==0) + (1 - info['rewards'][1]) * (self.actions==1)
					rewards_1         = info['rewards'][1] * (self.actions==1) + (1 - info['rewards'][0]) * (self.actions==0)
				elif leaky==-1:
					rewards_0         = info['rewards'][0]
					rewards_1         = info['rewards'][1]
				else:
					raise IndexError
				self.rewards = np.ascontiguousarray(np.concatenate((rewards_0[np.newaxis], rewards_1[np.newaxis])))
			else:
				raise ValueError('Expected a reward field in input dictionary')
			self.results = 0; self.T = self.actions.shape[0]; self.simulation = 0; self.trajectory = 0
			self.verify()
		else:
			self.load_results(info)

	def verify(self):
		'''
		Verifies some properties on the input data
		'''
		assert(np.all((self.rewards >= 0) * (self.rewards <= 1))), 'rewards not normalised'
		assert(np.all(np.unique(self.actions)==[0,1])), 'actions must be one or zero'
		assert(np.sum(self.idx_blocks)>0), 'the first trial at least must be reset to 0'
		if not (self.complete == 0 and self.leaky == -1): # particular case of 2 alphas in the complete condition, todo : take out?
			assert((self.leaky == -1) * (self.complete == 1) or (self.leaky != -1) * (self.complete == 0)), 'leaky can only be -1 in complete condition. In partial condition, the models to apply are leaky (1) (regress to mean) or counterfactual (0) (1 - r_obs)'

	def do_inference(self, noise = 1, apply_rep = 0, apply_weber = 1, condition = 1, curiosity_bias = 0, beta_softmax=-1, show_progress = False, temperature = True):
		'''
		---- Infencence method ----
		Takes as parameters:
			- noise = 1/0
			- apply_rep = 1/0
			- apply_weber = 1/0
			- condition = 1/0 : if noise = 1 , condition = observational_noise (noise in forced trials), if noise = 0, condition = apply_weber_decision_noise (weber-scaled softmax)
			- beta_softmax = -1/3 : softmax/argmax. If beta_softmax is set to 3, the value of the softmax parameter is 10**3 = 1000
			- temperature = temperature prior or beta prior. When inferring the beta, do we infer beta ~  U([0;100]) or T=1/beta ~ U([0;1]). 
							By default, we infer T=1/beta ~ U([0;1])
		This function generates the posterior of the parameters as well as the marginal likelihood (model evidence) estimator
		'''

		if curiosity_bias == 1 and apply_rep == 1:
			assert(False), 'curiosity_bias and apply_rep can not both be set to True'
		if curiosity_bias == 1:
			warnings.warn('With the curiosity_bias, no fictive rules are implemented')

		if noise==1:
			observational_noise = condition
			str_ofInterest      = 'observational_noise'
		elif noise==0:
			apply_weber_decision_noise = condition
			str_ofInterest             = 'apply_weber_decision_noise'
			if curiosity_bias == 1 and apply_weber_decision_noise == 1:
				assert(False), 'curiosity_bias and apply_weber_decision_noise can not both be set to True'			
		else:
			assert(False),'noise is {0}'.format(noise)
		self.inference_done = True

		print 'do inference: noise={0}, apply_rep={1}, apply_weber={2}, apply_guided={3}, beta_inferred={4}, beta_value={5}, curiosity_bias={9}, temperature_encoding={6}, {8}={7}'.format(noise, apply_rep, apply_weber, 0, beta_softmax==-1, beta_softmax, temperature, condition, str_ofInterest, curiosity_bias)
		self.traj_param = {'noise': noise, 'apply_rep': apply_rep, 'apply_weber': apply_weber, 'beta_softmax': beta_softmax, 'condition': condition}
		if self.complete == 1:
			if noise:
				self.results     = smc2_2q_1alpha.smc2(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, show_progress, apply_rep, apply_weber, beta_softmax, temperature, observational_noise)
				self.param_names = ['alpha', 'beta_softmax', 'epsilon'] + ['rep_bias'] * apply_rep
				self.idx_weights = -3
				self.idx_samples = 0
			else:
				if beta_softmax != -1:
					raise NotImplementedError
				else:
					self.results     = ibis_2q_1alpha.ibis(self.actions, self.rewards, self.choices, self.idx_blocks, self.subj_idx, apply_rep, apply_weber_decision_noise, curiosity_bias, show_progress, temperature)
					self.param_names = ['alpha', 'beta_softmax'] + ['rep_bias'] * apply_rep
					self.idx_weights = -3
					self.idx_samples = 0
		else:
			if noise:
				self.results     = smc2_2q_2alpha.smc2(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, show_progress, apply_rep, apply_weber, beta_softmax, temperature, observational_noise)
				self.param_names = ['alpha_chosen', 'alpha_unchosen', 'beta_softmax', 'epsilon'] + ['rep_bias'] * apply_rep
				self.idx_weights = -3
				self.idx_samples = 0
			else:
				if beta_softmax != -1:
					raise NotImplementedError
				else:
					self.results     = ibis_2q_2alpha.ibis(self.actions, self.rewards, self.choices, self.idx_blocks, self.subj_idx, apply_rep, apply_weber_decision_noise, curiosity_bias, show_progress, temperature)
					self.param_names = ['alpha_chosen', 'alpha_unchosen', 'beta_softmax'] + ['rep_bias'] * apply_rep
					self.idx_weights = -3
					self.idx_samples = 0

	# Get maximum a posteriori. Method called after the do_inference method.
	def get_map(self, temperature=True):
		'''
		Returns the maximum of posterior assuming normality of the posterior
		'''
		if (any(self.results[self.idx_weights] < 0) or (all(self.results[self.idx_weights]==0.))):
			if any(self.results[self.idx_weights] > 0):
				raise ValueError
			sample_weights = useful_functions.to_normalized_weights(self.results[self.idx_weights])
		else:
			sample_weights = self.results[self.idx_weights]
		self.map     = np.sum(self.results[self.idx_samples].T * sample_weights, axis=1)
		self.got_map = True
		if temperature:
			if self.complete:
				self.map[1] = np.sum(1./self.results[self.idx_samples][:,1].T * sample_weights)
			else:
				self.map[2] = np.sum(1./self.results[self.idx_samples][:,2].T * sample_weights)
		else:
			if self.complete:
				self.map[1] = np.sum((10**self.results[self.idx_samples][:,1].T) * sample_weights)
			else:
				self.map[2] = np.sum((10**self.results[self.idx_samples][:,2].T) * sample_weights)
		if temperature and self.traj_param['beta_softmax']==3:
			self.map[np.where([self.param_names[k] == 'beta_softmax' for k in range(len(self.param_names))])[0][0]] = 1000.
		print 'found map {0}'.format(self.map)

	# Get smoothing trajectory. A parameter sample must be mentioned to infer posteriori trajectories.
	def get_trajectory(self, use_default_param=True, **kwargs):
		'''
		Get smoothing trajectories: Q_{1:T}|a_{1:T} with Q_{1:T} the Q-values and a_{1:T} the subjects actions.
		Both the bootstrap and the guided filters are launched. By defaults, it uses the same parameters used in the inference method.
		If you want, you can change them with the kwargs argument but i do not advise it, it will perform some smoothing algorithm with parameters
		that do not correspond to the same model. However, i wanted to have that possibility.
		'''
		if self.traj_param['noise']==1:
			assert(self.traj_param['condition']==1), 'Smoothing trajectories are developed only when observational noise is assumed'
		else:
			assert(self.traj_param['condition']==0), 'Smoothing trajectories are developed only when decision noise is not weber'
		if use_default_param and self.got_map:
			sample = self.map
			for idx in range(len(self.traj_param.keys())):
				exec("%s = %d" % (self.traj_param.keys()[idx],self.traj_param[self.traj_param.keys()[idx]]))
		elif all([key in kwargs.keys() for key in ['noise', 'apply_rep', 'apply_weber', 'condition', 'sample']]):
			for idx in range(len(kwargs.keys())):
				exec("%s = %d" % (kwargs.keys()[idx],kwargs[kwargs.keys()[idx]]))
				self.traj_param[kwargs.keys()[idx]] = kwargs[kwargs.keys()[idx]]
		elif all([key in kwargs.keys() for key in ['noise', 'apply_rep', 'apply_weber', 'condition']]):
			sample = self.map
			for idx in range(len(kwargs.keys())):
				exec("%s = %d" % (kwargs.keys()[idx],kwargs[kwargs.keys()[idx]]))
				self.traj_param[kwargs.keys()[idx]] = kwargs[kwargs.keys()[idx]]
		elif 'sample' in kwargs.keys():
			sample = kwargs['sample']
			for idx in range(len(self.traj_param.keys())):
				exec("%s = %d" % (self.traj_param.keys()[idx],self.traj_param[self.traj_param.keys()[idx]]))
		else:
			raise ValueError('wrong arguments were given for function call')			

		if any([key in kwargs.keys() for key in ['noise', 'apply_rep', 'apply_weber', 'condition']]):
			for key in kwargs.keys():
				self.traj_param[key] = kwargs[key]
				exec("%s = %d" % (key,kwargs[key]))

		print 'get trajectory from sample = {4} with trajectory parameters noise={0}, apply_rep={1}, apply_weber={2}, condition={3}'.format(noise, apply_rep, apply_weber, self.traj_param['condition'], sample)
		if self.complete == 1:
			if noise:
				[backward_samples, margloglkd] = state_estimation_2q_1alpha.get_smoothing_trajectory(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, sample, apply_rep, apply_weber, 1)
				self.smoothing_guided          = backward_samples
				self.smoothing_mu_guided       = state_estimation_2q_1alpha.get_noiseless(self.rewards, self.idx_blocks, backward_samples, sample[0])
				[backward_samples, margloglkd] = state_estimation_2q_1alpha.get_smoothing_trajectory(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, sample, apply_rep, apply_weber, 0)
				self.smoothing_bootstrap       = backward_samples
				self.smoothing_mu_bootstrap    = state_estimation_2q_1alpha.get_noiseless(self.rewards, self.idx_blocks, backward_samples, sample[0])
				self.smoothing_param_lkd       = margloglkd
				self.smoothing                 = 0.
				if not self.param_names:
					self.param_names = ['alpha_chosen', 'beta_softmax', 'epsilon'] + ['rep_bias'] * apply_rep
			else:
				self.smoothing              = state_estimation_2q_1alpha.get_rl_traj(self.rewards, self.idx_blocks, sample[0])
				self.smoothing_mu           = 0.
				self.smoothing_guided       = 0.
				self.smoothing_mu_guided    = 0.
				self.smoothing_bootstrap    = 0.
				self.smoothing_mu_bootstrap = 0.
				if not self.param_names:
					self.param_names = ['alpha_chosen', 'beta_softmax'] + ['rep_bias'] * apply_rep
		else:
			if noise:
				[backward_samples, margloglkd] = state_estimation_2q_2alpha.get_smoothing_trajectory(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, sample, apply_rep, apply_weber, 1)
				self.smoothing_guided          = backward_samples
				self.smoothing_mu_guided       = state_estimation_2q_2alpha.get_noiseless(self.actions, self.rewards, self.idx_blocks, backward_samples, sample[0], sample[1])
				[backward_samples, margloglkd] = state_estimation_2q_2alpha.get_smoothing_trajectory(self.actions, self.rewards, self.idx_blocks, self.choices, self.subj_idx, sample, apply_rep, apply_weber, 0)
				self.smoothing_bootstrap       = backward_samples
				self.smoothing_mu_bootstrap    = state_estimation_2q_2alpha.get_noiseless(self.actions, self.rewards, self.idx_blocks, backward_samples, sample[0], sample[1])
				self.smoothing_param_lkd       = margloglkd
				self.smoothing                 = 0.
				if not self.param_names:
					self.param_names = ['alpha_chosen', 'alpha_unchosen', 'beta_softmax', 'epsilon'] + ['rep_bias'] * apply_rep
			else:
				self.smoothing              = state_estimation_2q_2alpha.get_rl_traj(self.actions, self.rewards, self.idx_blocks, sample[0], sample[1])
				self.smoothing_mu           = 0.
				self.smoothing_guided       = 0.
				self.smoothing_mu_guided    = 0.
				self.smoothing_bootstrap    = 0.
				self.smoothing_mu_bootstrap = 0.
				if not self.param_names:
					self.param_names = ['alpha_chosen', 'alpha_unchosen', 'beta_softmax'] + ['rep_bias'] * apply_rep		
		print('trajectory correctly found')
		self.got_traj = True

	def get_correlations(self):
		added_noise  = np.abs(self.smoothing[:,1:] - self.smoothing_mu[:,1:]) 
		update_noise = np.abs(self.smoothing[:,:-1] - self.smoothing_mu[:,1:])
		self.corr    = pearsonr(np.mean(added_noise, axis=0), np.mean(update_noise, axis=0))[0]
		print 'correlation between update and noise is {0}'.format(self.corr)

	def load_results(self, information):
		''' load results called in the constructor. As indicated up above, it loads all the items created by the save method'''
		self.inference_done          = information['inference_done']
		self.got_map                 = information['got_map']
		self.got_traj                = information['got_traj']
		self.actions                 = information['actions']
		self.choices                 = information['choices']
		self.idx_blocks              = information['blocks_idx']
		self.rewards                 = information['rewards']
		self.got_simul               = information['got_simul']
		self.complete                = information['complete']
		self.subj_idx                = information['subj_idx']
		self.leaky                   = information['leaky']
		if self.inference_done:
			self.results             = information['results']
			self.traj_param          = information['traj_param']
			self.param_names         = information['param_names']
			self.idx_weights         = information['idx_weights']
			self.idx_samples         = information['idx_samples']
		if self.got_map:
			self.map                 = information['map']
		if self.got_traj:
			self.smoothing           = information['traj_noiseless']
			self.smoothing_guided    = information['traj_noisy_guided']
			self.smoothing_mu_guided = information['traj_noisy_guided_means']
			self.smoothing_bootstrap = information['traj_noisy_bootstrap']
			self.smoothing_mu_bootstrap = information['traj_noisy_bootstrap_mean']
		if self.got_simul:
			self.simul_trajectories  = information['simul_traj']
			self.actions_simul       = information['simul_act']

	# simulate methods
	def simulate(self, use_default_param=True, true_rewards=None, forced_choices=None, **kwargs):
		'''
		Simulations
		'''
		if use_default_param and self.got_map:
			sample = self.map
			for idx in range(len(self.traj_param.keys())):
				exec("%s = %d" % (self.traj_param.keys()[idx],self.traj_param[self.traj_param.keys()[idx]]))
		elif all([key in kwargs.keys() for key in ['noise', 'apply_rep', 'apply_weber', 'sample']]):
			for idx in range(len(kwargs.keys())):
				exec("%s = %d" % (kwargs.keys()[idx],kwargs[kwargs.keys()[idx]]))
		elif all([key in kwargs.keys() for key in ['noise', 'apply_rep', 'apply_weber']]):
			sample = self.map
			for idx in range(len(kwargs.keys())):
				exec("%s = %d" % (kwargs.keys()[idx],kwargs[kwargs.keys()[idx]]))
		elif 'sample' in kwargs.keys():
			sample = kwargs['sample']
			for idx in range(len(self.traj_param.keys())):
				exec("%s = %d" % (self.traj_param.keys()[idx],self.traj_param[self.traj_param.keys()[idx]]))
		else:
			raise ValueError('wrong arguments were given for function call')
		
		# verifications
		if np.sum(self.choices==0) > 0 :
			assert(forced_choices is not None),'forced_choices argument must be given'

		print 'do simulation: noise={0}, apply_rep={1}, apply_weber={2}, condition={3}'.format(noise, apply_rep, apply_weber, condition)

		# get simulation
		if self.complete==0:
			print('sample {0}'.format(sample))
			if noise:
				self.simul_trajectories, self.actions_simul, self.rewards_simul = simulation.simulate_noisy_rl(true_rewards, self.complete, self.idx_blocks, self.choices, forced_choices, sample, self.traj_param['apply_rep'], self.traj_param['apply_weber'], self.traj_param['condition'])
			else:
				self.simul_trajectories, self.actions_simul, self.rewards_simul = simulation.simulate_noiseless_rl(true_rewards, self.complete, self.idx_blocks, self.choices, forced_choices, sample, apply_rep, condition)
		else:
			sample = np.concatenate((np.array([sample[0]]), sample))
			print('sample {0}'.format(sample))
			if noise:
				self.simul_trajectories, self.actions_simul, self.rewards_simul = simulation.simulate_noisy_rl(self.rewards, self.complete, self.idx_blocks, self.choices, forced_choices, sample, self.traj_param['apply_rep'], self.traj_param['apply_weber'], self.traj_param['condition'])
			else:
				self.simul_trajectories, self.actions_simul, self.rewards_simul = simulation.simulate_noiseless_rl(self.rewards, self.complete, self.idx_blocks, self.choices, forced_choices, sample, apply_rep, condition)		
		self.got_simul = 1

	def save_simulation(self, directory='simulations/', formats=['python']):
		if not self.got_simul:
			raise ValueError('No simulation was done')
		d = {}
		d['inference_done'] = 0
		d['got_map']        = 0
		d['got_traj']       = 0
		d['got_simul']      = 0
		d['actions']        = self.actions_simul
		d['rewards']        = self.rewards_simul
		d['choices']        = self.choices
		d['blocks_idx']     = self.idx_blocks
		d['complete']       = self.complete
		d['subj_idx']       = self.subj_idx
		d['leaky']          = self.leaky
		if self.got_map:
			d['param']      = self.map
		pram = ''
		for k in ['apply_weber', 'noise', 'apply_rep', 'condition']:
			pram += str(self.traj_param[k])
		pram += str(1*(self.traj_param['beta_softmax']==-1))
		path  = 'simul_2q_complete{0}_subj{1}_param{2}'.format(self.complete*1, self.subj_idx, pram)
		if self.leaky == 1:
			path += '_leaky1'
		elif self.leaky == 0:
			path += '_leaky0'
		if 'python' in formats:
			pkl.dump(d, open(directory + path + '.pkl', 'wb'))
		print('saved simulation. Path is {0}'.format(directory + path))

	def save(self, directory='results/', formats=['python', 'matlab']):
		'''
		save method, under python pickle dictionary or .mat format
		'''
		d = {}
		d['inference_done'] = self.inference_done
		d['got_map']        = self.got_map
		d['got_traj']       = self.got_traj
		d['got_simul']      = self.got_simul
		d['actions']        = self.actions
		d['rewards']        = self.rewards
		d['choices']        = self.choices
		d['blocks_idx']     = self.idx_blocks
		d['complete']       = self.complete
		d['subj_idx']       = self.subj_idx
		d['leaky']          = self.leaky
		if self.inference_done:
			d['results']     = self.results
			d['traj_param']  = self.traj_param
			d['param_names'] = self.param_names
			d['idx_weights'] = self.idx_weights
			d['idx_samples'] = self.idx_samples
		if self.got_map:
			d['map'] = self.map
		if self.got_traj:
			d['traj_noiseless']            = self.smoothing
			d['traj_noisy_guided']         = self.smoothing_guided
			d['traj_noisy_guided_means']   = self.smoothing_mu_guided
			d['traj_noisy_bootstrap']      = self.smoothing_bootstrap
			d['traj_noisy_bootstrap_mean'] = self.smoothing_mu_bootstrap
		if self.got_simul:
			d['simul_traj']    = self.simul_trajectories
			d['simul_act']     = self.actions_simul		
		pram = ''
		for k in ['apply_weber', 'noise', 'apply_rep', 'condition']:
			pram += str(self.traj_param[k])
		pram += str(1*(self.traj_param['beta_softmax']==-1))
		path  = '2q_complete{6}_subj{3}_resInf{0}_map{1}_traj{2}_simul{5}_param{4}'.format(self.inference_done*1, self.got_map*1, self.got_traj*1, self.subj_idx, pram, self.got_simul*1, self.complete*1)
		if self.complete == 0:
			path += '_leaky{0}'.format(self.leaky)
		if 'python' in formats:
			pkl.dump(d, open(directory + path + '.pkl', 'wb'))
		if 'matlab' in formats:
			savemat(directory + path, d)
		print 'saved under {0} format. Path is {1}'.format(formats, directory + path)








