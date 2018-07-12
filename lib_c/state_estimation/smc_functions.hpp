//
//  smc_functions.hpp
//  smc_wy
//
//  Created by Charles Findling on 10/11/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#ifndef smc_functions_hpp
#define smc_functions_hpp

#include <stdio.h>

namespace smc {

    void backward_smc_2q(double* traj_noisy, double* bw_noisy, double* weights, double alpha_c, double alpha_u, double epsilon, double* weights_unnorm, int* blocks, double* rewards, int* choices, int apply_weber, int T, int N_samples, int N_backward_samples);

    double smc_2q(double* Q_values, double* weights_unnorm, double* weights_res, double alpha_c, double alpha_u, double beta_softmax, double epsilon, int* ancestors_indexes, int N_samples, int T, int n_essay, int* blocks, int* actions, double* rewards, int* choices, int apply_rep_bias, int apply_weber, int apply_guided);

}


#endif /* smc_functions_hpp */
