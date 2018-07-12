//
//  smc_functions.cpp
//  smc_wy
//
//  Created by Charles Findling on 10/11/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#include "smc_functions.hpp"
#include <boost/random/mersenne_twister.hpp>
#include "../useful_functions/usefulFunctions.hpp"
#include "../useful_functions/usefulFunctions.cpp"
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <stdexcept>

namespace smc {
    
    using namespace std;
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::normal_distribution<> distribution(0., 1.);
    boost::math::normal s;

    double smc_2q(double* Q_values, double* weights_unnorm, double* weights_res, double alpha_c, double alpha_u, double beta_softmax, double epsilon, int* ancestors_indexes, int N_samples, int T, int n_essay, int* blocks, int* actions, double* rewards, int* choices, int apply_rep_bias, int apply_weber, int apply_guided)
    {
        generator.discard(40000);

        if (apply_rep_bias == 1)
        {
            throw std::logic_error("Repetition bias must be set to 0");
        }
        // Instantiate output vector and inner variables
        double marg_log_lkd = 0;
        int act             = -1;
        int ances_idx       = -1;
        int prev_act        = -1;
        double mu_0, mu_1, mu, noise_level_0, noise_level_1, mean_p, var_p, std_p, mean_p_0, var_p_0, prev_rew_0, prev_rew_1, Q0_anc, Q1_anc;
        double lambdaa = std::sqrt(3)/boost::math::constants::pi<double>();

        double weightsSum;

        // Loop over the time
        for (int t_idx = 0; t_idx < T; ++t_idx)
        {
            act        = *(actions + t_idx);
            weightsSum = 0;

            if ((*(blocks + t_idx)) == 0)
            {

                stratified_resampling(generator, weights_res, N_samples, ancestors_indexes);
                prev_rew_0 = *(rewards + t_idx - 1);
                prev_rew_1 = *(rewards + n_essay + t_idx - 1);

                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    ances_idx  = ancestors_indexes[n_idx];
                    Q0_anc     = (*(Q_values + 2 * n_essay * ances_idx + 2 * (t_idx - 1))); 
                    Q1_anc     = (*(Q_values + 2 * n_essay * ances_idx + 2 * (t_idx - 1) + 1)); 

                    if (actions[t_idx - 1] == 0)
                    {
                        mu_0     = (1 - alpha_c) * Q0_anc + alpha_c * prev_rew_0;
                        mu_1     = (1 - alpha_u) * Q1_anc + alpha_u * prev_rew_1;
                    }
                    else
                    {
                        mu_0     = (1 - alpha_u) * Q0_anc + alpha_u * prev_rew_0;
                        mu_1     = (1 - alpha_c) * Q1_anc + alpha_c * prev_rew_1;
                    }

                    if (apply_weber==1)
                    {
                        noise_level_0 = epsilon * std::abs(Q0_anc - prev_rew_0);
                        noise_level_1 = epsilon * std::abs(Q1_anc - prev_rew_1);
                    }
                    else
                    {
                        noise_level_0 = epsilon;
                        noise_level_1 = epsilon;
                    }

                    *(Q_values + 2 * n_idx * n_essay + 2 * t_idx) = distribution(generator) * noise_level_0 + mu_0;

                    if (apply_guided&&(choices[t_idx]==1)&&(noise_level_1>0))
                    {
                        mu = *(Q_values + 2 * n_idx * n_essay + 2 * t_idx) - mu_1;
                        if (act==1)
                        {
                            mean_p = -moment_proposal(s, 1, -mu, noise_level_1/2., lambdaa, beta_softmax);
                            var_p  = moment_proposal(s, 2, -mu, noise_level_1/2., lambdaa, beta_softmax) - std::pow(mean_p, 2);
                        }
                        else
                        {
                            mean_p = moment_proposal(s, 1, mu, noise_level_1/2., lambdaa, beta_softmax);
                            var_p  = moment_proposal(s, 2, mu, noise_level_1/2., lambdaa, beta_softmax) - std::pow(mean_p, 2);                            
                        }
                        if ((var_p > 0)&&(isinf(var_p)==0))
                        {
                            std_p = std::sqrt(var_p);
                        }
                         else
                        {
                            std_p = noise_level_1;
                        }
                        mean_p                                      = *(Q_values + 2 * n_idx * T + 2 * t_idx) - mean_p;
                        *(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1) = distribution(generator) * std_p + mean_p;
                    }
                    else
                    {
                        *(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1) = distribution(generator) * noise_level_1 + mu_1;
                    }

                    if ((choices[t_idx]==1)&&apply_guided&&(noise_level_1>0))
                    {
                        *(weights_res + n_idx)  = log_logistic_proba(beta_softmax, *(Q_values + 2 * n_idx * n_essay + 2 * t_idx), *(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1), act) + log_calculate_weight_p(*(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1), mu_1, noise_level_1, mean_p, std_p); 
                    }
                    else if (choices[t_idx]==1)
                    {
                        *(weights_res + n_idx)  = log_logistic_proba(beta_softmax, *(Q_values + 2 * n_idx * n_essay + 2 * t_idx), *(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1), act);
                    }
                    else
                    {
                        *(weights_res + n_idx) = 0.;
                    }
                    // weights
                    *(weights_unnorm + n_essay * n_idx + t_idx) = *(weights_res + n_idx);
                }
            }
            else
            {
                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    *(Q_values + 2 * n_idx * n_essay + 2 * t_idx)         = 1./2;
                    *(Q_values + 2 * n_idx * n_essay + 2 * t_idx + 1)     = 1./2;

                    // weights
                    if (choices[t_idx]==1)
                    {
                        *(weights_res + n_idx) = std::log(.5);
                    }
                    else
                    {
                        *(weights_res + n_idx) = 0.; 
                    }

                    *(weights_unnorm + n_essay * n_idx + t_idx) = *(weights_res + n_idx);
                }
            }

            double b   = *max_element(weights_res, weights_res + N_samples);
            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_res + n_idx)                      = exp(*(weights_res + n_idx) - b);
                *(weights_unnorm + n_essay * n_idx + t_idx) = exp(*(weights_unnorm + n_essay * n_idx + t_idx) - b);
                weightsSum                                 += *(weights_res + n_idx);
            }
            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_res + n_idx)                      = *(weights_res + n_idx)/weightsSum;                      
                *(weights_unnorm + n_essay * n_idx + t_idx) = *(weights_unnorm + n_essay * n_idx + t_idx)/weightsSum;
            }
            marg_log_lkd += (b + log(weightsSum) - log(N_samples));
        }
        return marg_log_lkd;
    }

    void backward_smc_2q(double* traj_noisy, double* bw_noisy, double* weights, double alpha_c, double alpha_u, double epsilon, double* weights_unnorm, int* blocks, double* rewards, int* actions, int apply_weber, int T, int N_samples, int N_backward_samples)
    {
        //variables
        vector<double> prev_rew(2, -1);
        double mu_0, mu_1, noise_level_0, noise_level_1;
        boost::math::normal s;
        int idx;
        int n_essay = T;
        double sum_weights = 0;
        for (int n_idx = 0; n_idx < N_samples; ++n_idx) {
            *(weights + n_idx) = *(weights_unnorm + n_idx * T + T - 1);
            sum_weights = sum_weights + *(weights + n_idx);
        }
        divide(weights, sum_weights, N_samples);
        
        for (int b_idx = 0; b_idx < N_backward_samples; ++b_idx)
        {
            idx = Sample_Discrete_Distribution(generator, weights, N_samples);
            *(bw_noisy + 2 * n_essay * b_idx + 2 * (T - 1))     = *(traj_noisy + 2 * n_essay * idx + 2 * (T - 1));
            *(bw_noisy + 2 * n_essay * b_idx + 2 * (T - 1) + 1) = *(traj_noisy + 2 * n_essay * idx + 2 * (T - 1) + 1);
        }
        
        for (int t_idx = T - 2; t_idx >= 0; --t_idx)
        {
            if ((*(blocks + t_idx + 1)) == 0)
            {
                prev_rew[0] = *(rewards + t_idx);
                prev_rew[1] = *(rewards + n_essay + t_idx);
                
                for (int b_idx = 0; b_idx < N_backward_samples; ++b_idx)
                {
                    sum_weights = 0.;
                    for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                    {
                        if (actions[t_idx - 1]==0)
                        {
                            mu_0     = (1 - alpha_c) * (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx)) + alpha_c * prev_rew[0];
                            mu_1     = (1 - alpha_u) * (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx + 1)) + alpha_u * prev_rew[1];
                        }
                        else
                        {
                            mu_0     = (1 - alpha_u) * (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx)) + alpha_u * prev_rew[0];
                            mu_1     = (1 - alpha_c) * (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx + 1)) + alpha_c * prev_rew[1];
                        }

                        if (apply_weber==1)
                        {
                            noise_level_0 = epsilon * std::abs(prev_rew[0] - (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx)));
                            noise_level_1 = epsilon * std::abs(prev_rew[1] - (*(traj_noisy + n_idx * n_essay * 2 + 2 * t_idx + 1)));
                        }
                        else
                        {
                            noise_level_0 = epsilon;
                            noise_level_1 = epsilon;
                        }

                        if ((noise_level_1>0)&&(noise_level_0>0))
                        {
                            *(weights + n_idx) = normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2), mu_0, noise_level_0) * normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2 + 1), mu_1, noise_level_1) * (*(weights_unnorm + n_idx * n_essay + t_idx));
                        }
                        else if (noise_level_0 > 0)
                        {
                            *(weights + n_idx) = normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2), mu_0, noise_level_0) * (*(bw_noisy + 2 * n_essay * b_idx + 2 * (t_idx + 1) + 1) == mu_1) * (*(weights_unnorm + n_idx * n_essay + t_idx));
                        }
                        else if (noise_level_1 > 0)
                        {
                            *(weights + n_idx) = (*(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2) == mu_0) * normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2 + 1), mu_1, noise_level_1) * (*(weights_unnorm + n_idx * n_essay + t_idx));
                        }
                        else
                        {
                            *(weights + n_idx) = (*(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2) == mu_0) * (*(bw_noisy + 2 * n_essay * b_idx + 2 * (t_idx + 1) + 1) == mu_1) * (*(weights_unnorm + n_idx * n_essay + t_idx));
                        }
                        sum_weights        = sum_weights + (*(weights + n_idx));
                    }
                    if (sum_weights == 0)
                    {
                        std::cout << *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2) << std::endl;
                        std::cout << mu_0 << std::endl;
                        std::cout << noise_level_0 << std::endl;
                        std::cout << normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2), mu_0, noise_level_0) << std::endl;
                        std::cout << normal_pdf(s, *(bw_noisy + 2 * n_essay * b_idx + (t_idx + 1) * 2 + 1), mu_1, noise_level_1) << std::endl;
                        std::cout << (*(weights_unnorm + (N_samples - 1) * n_essay + t_idx)) << std::endl;
                        std::cout << t_idx << std::endl;
                        std::cout << noise_level_1 << std::endl;
                        std::cout << noise_level_0 << std::endl;
                        throw std::logic_error( "sum of weights is 0" );
                    }
                    divide(weights, sum_weights, N_samples);
                    idx = Sample_Discrete_Distribution(generator, weights, N_samples);
                    *(bw_noisy + 2 * b_idx * n_essay + 2 * t_idx)     = *(traj_noisy + 2 * n_essay * idx + 2 * t_idx);
                    *(bw_noisy + 2 * b_idx * n_essay + 2 * t_idx + 1) = *(traj_noisy + 2 * n_essay * idx + 2 * t_idx + 1);
                }
            }
            else
            {
                sum_weights = 0.;
                for (int n_idx = 0; n_idx < N_samples; ++n_idx) {
                    *(weights + n_idx) = *(weights_unnorm + n_idx * n_essay + t_idx);
                    sum_weights       += *(weights + n_idx);
                }
                divide(weights, sum_weights, N_samples);

                for (int b_idx = 0; b_idx < N_backward_samples; ++b_idx)
                {
                    idx = Sample_Discrete_Distribution(generator, weights, N_samples);
                    *(bw_noisy + 2 * b_idx * n_essay + 2 * t_idx)     = *(traj_noisy + 2 * n_essay * idx + 2 * t_idx);
                    *(bw_noisy + 2 * b_idx * n_essay + 2 * t_idx + 1) = *(traj_noisy + 2 * n_essay * idx + 2 * t_idx + 1);
                }
            }
        }
        return;
    }
}
















