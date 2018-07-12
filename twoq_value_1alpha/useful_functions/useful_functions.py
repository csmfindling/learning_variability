##### Useful functions used in the script #####

# Libraries
import numpy as np
import math
import operator
import functools
from scipy.stats import beta as betalib
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import gamma, norm, invgamma

# Gamma pdf with parameters alpha, and an intensity parameter beta
def sample_gamma(alpha, beta, numberOfSamples=1, plot=False):
    if plot:
        plt.hist(gamma.rvs(alpha, size = numberOfSamples)*(1./beta),bins=100);
    return gamma.rvs(alpha, size = numberOfSamples)*(1./beta);

def sample_invgamma(alpha, beta, numberOfSamples=1, plot=False):
    if plot:
        plt.hist(gamma.rvs(alpha, size = numberOfSamples) * beta,bins=100);
    return invgamma.rvs(alpha, size = numberOfSamples) * beta;


# Truncated gaussian
def sample_truncated_gaussian(*args):
    if len(args) == 3:
        mu = args[0]; std = args[1]; minimum = args[2]
        while 1:
            sample = np.random.normal(mu, std)
            if sample > minimum:
                return sample
    elif len(args) == 4:
        mu = args[0]; std = args[1]; minimum = args[2]; maximum = args[3]
        while 1:
            sample = np.random.normal(mu, std)
            if (sample > minimum) and (sample < maximum):
                return sample
    else:
        raise Exception('invalid number of arguments')

def get_ess(weights_norm):
    return 1./sum(weights_norm**2)

def logistic_proba(beta, particle, action):
    value = 1./(1. + np.exp(beta*(particle[0] - particle[1])))
    return (value**action) * ((1 - value)**(1 - action))

# Stratified resample:
def stratified_resampling(w):
    N = len(w)
    v = np.cumsum(w) * N
    s = np.random.uniform()
    o = np.zeros(N, dtype=np.int)
    m = 0
    for i in range(N):
        while v[m] < s : m = m + 1
        o[i] = m; s = s + 1
    return o

# Multinomial un-normalized pick function; the un-normalized probability distribution is p.
def random_pick(p):
	return np.random.choice(len(p), p = np.divide(p,np.sum(p)))

def random_pick_list(p,n):
    return np.random.choice(len(p), size = n, p = np.divide(p,np.sum(p)))

def dirichlet_pdf(x, alpha):
  return (math.gamma(sum(alpha)) / 
          functools.reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          functools.reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))

def log_dirichlet_pdf(x,alpha):
    return (gammaln(sum(alpha)) - functools.reduce(operator.add, [gammaln(a) for a in alpha]) + functools.reduce(operator.add, [(alpha[i] - 1)*np.log(x[i]) for i in range(len(alpha))]))

def log_beta_pdf(x, a, b):
    return gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1)*np.log(x) + (b - 1)*np.log(1 - x)

def log_sum(logvector):
    b = np.max(logvector)
    return b + np.log(functools.reduce(operator.add, [np.exp(logw - b) for logw in logvector]))

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

def autocorrelation(x):
    x_norm = np.divide(x - np.mean(x),np.std(x))
    result = np.correlate(np.divide(x_norm, len(x)), x_norm, mode='full')
    return result[result.size/2:]

def plot_results(td, Z_prob, last_Z_prob, tau_params, beta_params, gamma_params, m_h_tau, m_h_gamma, A_corr_count, tau_autocorrelation):
    [trial_num, K] = gamma_params.shape; Z_true = td['Z'];
    sample_num     = len(tau_autocorrelation);
    plt.figure(figsize=(12, 9));

    # Plot beta
    plt.subplot(2,2,1);
    beta_mean = np.divide(beta_params[:,0], np.sum(beta_params,axis=1));
    beta_std  = np.sqrt(np.divide(np.multiply(beta_mean, beta_params[:,1]), np.multiply(np.sum(beta_params, axis=1), np.sum(beta_params, axis=1)+1)))
    plt.plot(beta_mean, 'r-');plt.hold(True); plt.fill_between(np.arange(trial_num),beta_mean-beta_std, beta_mean+beta_std,facecolor=[1,.5,.5], color=[1,.5,.5]); 
    plt.axis([0,trial_num-1, 0, 1 ]);
    switch_trials = np.where(td['B'])[0];
    plt.plot([0, trial_num], [td['beta'], td['beta']], 'r--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated beta parameters'); 

    # Plot tau
    plt.subplot(2,2,3);
    tau_mean = np.divide(tau_params[:,0], np.sum(tau_params, axis=1));
    tau_std  = np.sqrt(np.divide(np.multiply(tau_mean, tau_params[:,1]), np.multiply(np.sum(tau_params, axis=1), np.sum(tau_params, axis=1)+1)));
    plt.plot(tau_mean, 'b-');plt.hold(True); plt.fill_between(np.arange(trial_num), tau_mean - tau_std, tau_mean+tau_std, facecolor=[.5,.5,1],color = [.5,.5,1]); 
    plt.axis([0, trial_num-1, 0, 1]);
    plt.plot([0, trial_num], [td['tau'], td['tau']], 'b--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated tau paramaters'); 

    # Plot gamma paramaters
    plt.subplot(2,2,4);
    plt.imshow(gamma_params.T); plt.hold(True);
    plt.plot(Z_true, 'k--', linewidth=1);
    plt.axis([0,trial_num-1, 0, K-1]);
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('Estimated gamma parameters');

    # Plot state probability
    plt.subplot(2,2,2);
    plt.imshow(Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1])
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('p(TS|past) at decision time');

    plt.draw();

    # Plot performances
    plt.figure(figsize=(12,9));

    #plot final performance
    plt.subplot(2,2,1)
    plt.plot(np.divide(A_corr_count, np.arange(trial_num)+1), 'k-', linewidth=2); plt.hold(True);
    plt.axis([0,trial_num-1,0,1]);
    plt.hold(False)
    plt.xlabel('trials');
    plt.ylabel('proportion correct answers');

    #plot final Z estimated
    plt.subplot(2,2,3);
    plt.imshow(last_Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1]);
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('p(TS|past) at current time');

    #plot gamma metropolis-hasting acceptance rate
    plt.subplot(2,2,2);
    plt.plot(m_h_gamma, 'g-');plt.hold(True);
    plt.plot(m_h_tau, 'b-');
    plt.axis([0,trial_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('gamma(green)/tau(blue) acceptance rates');

    #plot gibbs autocorrelation function
    plt.subplot(2,2,4);
    plt.plot(tau_autocorrelation, 'k-'); plt.hold(True);
    plt.axis([0,sample_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('Gibbs sampler autocorrelation');


    plt.draw();

    return 'plot ok'