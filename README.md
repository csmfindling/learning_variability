<h1> Computational variability in reinforcement learning drives exploratory decisions in volatile environments </h1>

This is a project realised at Ecole Normale Supérieure by Vasilisa Skvortsova and Charles Findling under the supervisions of Valentin Wyart and Stefano Palminteri with collaborator Remi Dromnelle.

<h3> Link to the paper </h3>

Briefly, the paper investigates the presence of variability in learning and studies its impact on behavior through computational modelling and neuroimaging procedures.

<h3> Summary of the code </h3>

This code offers flexible implementations in the case of Q-learning algorithms. Four big types of Q-learning algorithms are developped, obtained with combining noiseless and noisy algorithms with 1 or 2 Q-values. 

General arhitecture of the code:
* smc_object.py - *main file, creates an object which gives access to all inference and smoothing functions*
* lib_c - *C files*
  * smc2 - *C files for all smc2 procedures*
  * state_estimation - *C files for all smoothing procedures*
  * useful_functions - *file for all used C auxiliary functions*
* twoq_value_1alpha - *python files for smc procedures in the case of a unique learning rate*
  * ibis - *python files for smc procedures in the case of noise-free RL*
  * smc2 - *python files for smc procedures in the case of noisy RL*
  * state_estimation - *python files for smoothing procedures*
  * useful_functions - *python files with all used auxiliary functions*
* twoq_value_2alpha - *python files for smc procedures in the case of distinct learning rates*
  * ibis - *python files for smc procedures in the case of noise-free RL*
  * smc2 - *python files for smc procedures in the case of noisy RL*
  * state_estimation - *python files for smoothing procedures*
  * useful_functions - *python files with all used auxiliary functions*
  * simulation - *python files with simulation procedures. Attention : this file is used for simulating also the one learning rate case.

<h3> Code compilation </h3>

To compile the c++ libraries, you will need to install the boost c++ library, version 1.59 - https://www.boost.org/users/history/version_1_59_0.html

Once downloaded, open the compile_c.sh file. Modify it by adding your boost library path. Then launch ./compile_c.sh.


<h3> Code instructions </h3>

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
         
The first thing you will want to do is create an smc object. The constructor is the parameters:
1. A dictionary with keys
 1. *actions*, of shape (T) speciying the actions of the subject - (0 or 1). T is the total number of trials
 1. *rewards*, of shape (2, T) with T the length of the experiment. These should be normalised between 0 and 1. In the partial case, the learning rule of the unchosen option will override the unchosen reward.
 1. *subject_idx*, an integer speciying the index of the subject. This is for saving purposes. By default, it will be 0
 1. *choices* , of shape (T) speciying whether the trials was a choice or a forced trial. By default, it will be np.ones(T), assuming thus there are no forced trials
 1. *blocks_idx*, of shape (T), specifying the beginning of each blocks. If it is the beginning of a new block, a 1 should be present. By default, it will be set to blocks_idx = np.zeros(T), with blocks_idx[0]= 1, assuming thus only one block.
1. A *complete* argument in (1/0) specifying whether you are in the complete or feedback setting
1. A *leaky* argument in (-1/0/1) specifying the learning rule in the partial case. When complete, there is nothing to do, leaky=-1. In partial, one must choose between the leaky model or anticorrelated model leaky = 1/0; if leaky == 1, then the regression to the mean model will be applied; elif leaky == 0 then the 1 - R model will the applied else an error will be raise
1. A *onload* argument = True/False : If onload is set to True, the dictionary input expects much more variables (see load_results function). Essentially, it expects the contents of the output of the save function

A simple example is given below. First, import the smc object file

```
from smc_object import smc_object  
```
Then, create an smc_object by specifying a path and the complete/partial argument
```
p = 'subj2_exp2.pkl'  
s_obj = smc_object(path = p, exp_idx = 2, complete = 1)
```
Do inference by specifying the wanted parameters. Then get MAP and smoothing trajectory
```
s_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, apply_guided=1, beta_softmax=-1, show_progress=True)  
s_obj.get_map() 
s_obj.get_trajectory()  
```
Plot and save results
```
s_obj.plot()  
s_obj.save(directory='results/') 
```

<h3> A pure python code </h3>

A python-only code to implement inference in RL procedures with learning variability will be accessible shortly.

<h3> Enquiries </h3>

If you have any questions on this code or related, please do not hesitate to contact me: charles.findling(at)gmail.com

<h3> References </h3>



