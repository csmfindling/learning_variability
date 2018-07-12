//
//  usefulFunctions.cpp
//  CreateTask
//
//  Created by Charles Findling on 22/09/2015.
//  Copyright © 2015 Charles Findling. All rights reserved.
//

#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <numeric>
#include <cmath>
#include "usefulFunctions.hpp"
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>

using namespace std;

double moment_proposal(boost::math::normal &s, int k, double mu, double epsilon, double lambdaa, double beta)
{
    const double pi = boost::math::constants::pi<double>();
    
    double Z = cdf(s, (lambdaa * beta * mu)/std::sqrt(1 + 4 * std::pow(epsilon, 2) * std::pow(lambdaa, 2) * std::pow(beta, 2)));
    if (Z==0.)
    {
        if (k==1)
        {
            return mu;
        }
        else
        {
            return -10;
        }
        
    }
    if (k==0)
    {
        return Z;
    }
    else if (k==1)
    {
        double t = std::sqrt(1 + 4 * std::pow(epsilon,2) * std::pow(lambdaa,2) * std::pow(beta,2));
        return 4 * std::pow(epsilon,2) * lambdaa * beta / t * pdf(s, mu * lambdaa * beta / t) / Z + mu;
    }
    else if (k==2)
    {
        double a = lambdaa * beta * mu;
        double b = lambdaa * beta * 2 * epsilon;
        double t = std::sqrt(1 + std::pow(b,2));
        double value  = 4 * std::pow(epsilon,2) / Z * (1 - cdf(s, -mu/(2*epsilon * std::sqrt(1 + 1./std::pow(b,2)))) - 1./(t * std::sqrt(2 * pi)) * a * std::pow(b,2)/(1+std::pow(b,2)) * std::exp((std::pow(a,2) * std::pow(b,2)/(2*(1+std::pow(b,2))) - (std::pow(a,2))/2)));
        value += 4 * mu * epsilon/Z * b/t * pdf(s, a/t) + std::pow(mu,2);
        return value;
    }
    else
    {
        return 0.;
    }
}

double moment_proposal(boost::math::normal &s, int k, double mu, double epsilon, double lambdaa, double beta, double repetition)
{
    const double pi = boost::math::constants::pi<double>();
    
    double Z = cdf(s, (lambdaa * beta * mu + lambdaa * repetition)/std::sqrt(1 + 4 * std::pow(epsilon, 2) * std::pow(lambdaa, 2) * std::pow(beta, 2)));
    if (Z==0.)
    {
        if (k==1)
        {
            return mu;
        }
        else
        {
            return -10;
        }
        
    }
    if (k==0)
    {
        return Z;
    }
    else if (k==1)
    {
        double t = std::sqrt(1 + 4 * std::pow(epsilon,2) * std::pow(lambdaa,2) * std::pow(beta,2));
        return 4 * std::pow(epsilon,2) * lambdaa * beta / t * pdf(s, (lambdaa * beta * mu + lambdaa * repetition) / t) / Z + mu;
    }
    else if (k==2)
    {
        double a = lambdaa * beta * mu + lambdaa * repetition;
        double b = lambdaa * beta * 2 * epsilon;
        double t = std::sqrt(1 + std::pow(b,2));
        double value  = 4 * std::pow(epsilon,2) / Z * (1 - cdf(s, -a/(b * std::sqrt(1 + 1./std::pow(b,2)))) - 1./(t * std::sqrt(2 * pi)) * a * std::pow(b,2)/(1+std::pow(b,2)) * std::exp((std::pow(a,2) * std::pow(b,2)/(2*(1+std::pow(b,2))) - (std::pow(a,2))/2)));
        value += 4 * mu * epsilon/Z * b/t * pdf(s, a/t) + std::pow(mu,2);
        return value;
    }
    else
    {
        return 0.;
    }
}

double logistic(double Q_0, double Q_1, double beta)
{
    return 1./(1 + exp(beta * (Q_0 - Q_1)));
}

double calculate_weight_precision(double a_0, double b_0, double a_0_p, double b_0_p, double Q_0, double a_1, double b_1, double a_1_p, double b_1_p, double Q_1, double beta, int act)
{
    double value = 1./(1 + exp(beta * (Q_0 - Q_1)));
    return std::pow(value, act) * std::pow(1 - value,1 - act) * boost::math::ibeta_derivative(a_0, b_0, Q_0) * boost::math::ibeta_derivative(a_1, b_1, Q_1) / (boost::math::ibeta_derivative(a_0_p, b_0_p, Q_0) * boost::math::ibeta_derivative(a_1_p, b_1_p, Q_1)); 
}

double calculate_weight_p(boost::math::normal s, double Q_1, double Q_1_mu, double epsilon, double mu_1_p, double std_1_p)
{
    return pdf(s, (Q_1 - Q_1_mu)/epsilon) * (std_1_p/epsilon) /pdf(s, (Q_1 - mu_1_p)/std_1_p);
}

double log_calculate_weight_p(double Q_1, double Q_1_mu, double epsilon, double mu_1_p, double std_1_p)
{
    return -.5 * std::pow((Q_1 - Q_1_mu)/epsilon, 2.) + std::log(std_1_p/epsilon) + .5 * std::pow((Q_1 - mu_1_p)/std_1_p, 2.);
}

double log_logistic_proba(double beta, double Q_0, double Q_1, int action)
{
    double b           = std::max(0., beta * (Q_0 - Q_1));
    double log_value_1 =  - (b + log(exp(0. - b) + exp(beta * (Q_0 - Q_1) - b))); //-std::log(1 + exp(beta * (Q_0 - Q_1)));
    b                  = std::max(0., -beta * (Q_0 - Q_1));
    double log_value_0 = - (b + log(exp(0. - b) + exp(- beta * (Q_0 - Q_1) - b)));
    return action * log_value_1 + (1. - action) * log_value_0;
}

double log_sum(vector<double> logvector){
    double b          = *max_element(logvector.begin(), logvector.end());
    double res        = 0;
    unsigned long numberOfElements = logvector.size();
    for (int i = 0; i != numberOfElements; ++i) {
        res = res + exp(logvector[i] - b);
    }
    return b + log(res);
}

/*double log_logistic_proba(double beta, double dQ, int action)
{
    double value = 1./(1. + exp(beta * dQ));
    return action * std::log(value) + (1 - action) * std::log(1 - value);
}
*/

double log_logistic_proba(double beta, double dQ, int action)
{
    double b           = std::max(0., beta * dQ);
    double log_value_1 =  - (b + log(exp(0. - b) + exp(beta * dQ - b))); //-std::log(1 + exp(beta * (Q_0 - Q_1)));
    b                  = std::max(0., -beta * dQ);
    double log_value_0 = - (b + log(exp(0. - b) + exp(- beta * dQ - b)));
    return action * log_value_1 + (1 - action) * log_value_0;
}

double log_logistic_proba(double beta, double repetition, double dQ, int action, int prev_action)
{
    double b           = std::max(0., beta * dQ - sgn(prev_action - .5) * repetition);
    double log_value_1 = - (b + log(exp(0. - b) + exp(beta * dQ - sgn(prev_action - .5) * repetition - b)));
    b                  = std::max(0.,  - beta * dQ + sgn(prev_action - .5) * repetition);
    double log_value_0 = - (b + log(exp(0. - b) + exp(- beta * dQ + sgn(prev_action - .5) * repetition - b)));
    // double value = 1./(1 + exp(beta * dQ - sgn(prev_action - .5) * repetition));
    return action * log_value_1 + (1 - action) * log_value_0;
}

double log_logistic_proba(double beta, double repetition, double Q_0, double Q_1, int action, int prev_action)
{
    double dQ          = Q_0 - Q_1;
    double b           = std::max(0., beta * dQ - sgn(prev_action - .5) * repetition);
    double log_value_1 = - (b + log(exp(0. - b) + exp(beta * dQ - sgn(prev_action - .5) * repetition - b)));
    b                  = std::max(0.,  - beta * dQ + sgn(prev_action - .5) * repetition);
    double log_value_0 = - (b + log(exp(0. - b) + exp(- beta * dQ + sgn(prev_action - .5) * repetition - b)));
    return action * log_value_1 + (1 - action) * log_value_0;
}

double calculate_weight(double Q_0, double Q_1, double Q_1_mu, double epsilon, double a, double b, int act)
{
    if (act == 0)
    {
        if (Q_0 > Q_1)
        {
            if (Q_1_mu < Q_0)
            {
                return boost::math::ibeta(a, b, Q_0);
            }
            else
            {
                double a_0 = std::max(1., Q_1_mu/epsilon);
                double b_0 = std::max(1., (1 - Q_1_mu)/epsilon);
                return boost::math::ibeta(a, b, Q_0) * boost::math::ibeta_derivative(a_0, b_0, Q_1) / boost::math::ibeta_derivative(a, b, Q_1) ; //cdf(s, (Q_0 - Q_1_mu)/std) - cdf(s, - Q_1_mu/std);
            }
        }
        else
        {
            return 0;
        }
    }
    else
    {
        if (Q_1 > Q_0)
        {
            if (Q_1_mu > Q_0)
            {
                return 1 - boost::math::ibeta(a, b, Q_0); 
            }
            else
            {
                double a_0 = std::max(1., Q_1_mu/epsilon);
                double b_0 = std::max(1., (1 - Q_1_mu)/epsilon);
                return (1 - boost::math::ibeta(a, b, Q_0)) * boost::math::ibeta_derivative(a_0, b_0, Q_1) / boost::math::ibeta_derivative(a, b, Q_1) ;  //cdf(s, (1 - Q_1_mu)/std) - cdf(s, (Q_0 - Q_1_mu)/std);
            }
        }
        else
        {
            return 0;
        }
    }
}

double normal_pdf(boost::math::normal &s, double x, double mu, double std)
{
    return pdf(s, (x - mu)/std) * (1./std);
}

double log_normal_pdf(double x, double mu, double std)
{
    const double pi = boost::math::constants::pi<double>();
    return - .5 * std::pow((x - mu)/std, 2) - std::log(std * std::sqrt(2 * pi));
}

double truncated_normal_pdf(double x, double mu, double std, double min, double max)
{
    boost::math::normal s;
    return pdf(s, (x - mu)/std) * (1./std)/(cdf(s, (max - mu)/std) - cdf(s, (min - mu)/std) ) * (x >= 0) * (x <= 1);
}

double truncated_normal_pdf(boost::math::normal s, double x, double mu, double std, double min, double max)
{
    return pdf(s, (x - mu)/std) * (1./std)/(cdf(s, (max - mu)/std) - cdf(s, (min - mu)/std) ) * (x >= 0) * (x <= 1) ;
}


double logistic_proba(double beta, double Q_0, double Q_1, int action)
{
    double value = 1./(1 + exp(beta * (Q_0 - Q_1)));
    return std::pow(value, action) * std::pow(1 - value,1 - action);
}

double logistic_proba(double beta, double dQ, int action)
{
    double value = 1./(1. + exp(beta * dQ));
    return std::pow(value, action) * std::pow(1 - value,1 - action);
}

double softmax_proba(double beta, double Q_0, double Q_1, double Q_2, double Q_3, int action)
{
    if (action == 0)
    {
        return exp(beta * Q_0)/(exp(beta * Q_0) + exp(beta * Q_1) + exp(beta * Q_2) + exp(beta * Q_3));
    }
    else if (action == 1)
    {
        return exp(beta * Q_1)/(exp(beta * Q_0) + exp(beta * Q_1) + exp(beta * Q_2) + exp(beta * Q_3));
    }
    else if (action == 2)
    {
        return exp(beta * Q_2)/(exp(beta * Q_0) + exp(beta * Q_1) + exp(beta * Q_2) + exp(beta * Q_3));
    }
    else
    {
        return exp(beta * Q_3)/(exp(beta * Q_0) + exp(beta * Q_1) + exp(beta * Q_2) + exp(beta * Q_3));
    }
}

double logistic_proba(double beta, double repetition, double Q_0, double Q_1, int action, int prev_action)
{
    double value = 1./(1 + exp(beta * (Q_0 - Q_1) - sgn(prev_action - .5) * repetition));
    return std::pow(value, action) * std::pow(1 - value,1 - action);
}

double logistic_proba(double beta, double repetition, double dQ, int action, int prev_action)
{
    double value = 1./(1 + exp(beta * dQ - sgn(prev_action - .5) * repetition));
    return std::pow(value, action) * std::pow(1 - value,1 - action);
}

double exp_log(double sum_so_far, double x)
{
    return sum_so_far + exp(x);
}

double log_sum(double* logvector, int numberOfElements){
    double b          = *max_element(logvector, logvector + numberOfElements);
    double res        = 0;
    for (int i = 0; i!=numberOfElements; ++i) {
        res = res + exp(logvector[i] - b);
    }
    return b + log(res);
}

vector<double> to_normalized_weights(vector<double> &logvector){
    vector<double> res(logvector.size());
    double b          = *max_element(logvector.begin(), logvector.end());
    for (int i = 0; i != logvector.size(); ++i) {
        res[i] = exp(logvector[i] - b);
    }
    res = res/sum(res);
    return res;
}

/*void to_normalized_weights(double* logvector, int size){
    double b          = *max_element(logvector, logvector + size);
    for (int i = 0; i != size; ++i) {
        *(logvector + i) = exp(*(logvector + i) - b);
    }
    divide(logvector, size)
    return;
}
*/
template<typename T>
std::vector<T> maximum(std::vector<T> &vect, T element)
{
    unsigned long dim = vect.size();
    std::vector<T> res(dim);
    for (int i = 0; i != dim; ++i){
        res[i] = std::max(vect[i], element);
    }
    return res;
}

double beta_function(std::vector<double> alpha)
{
    return exp(log_beta_function(alpha));
}

double beta_function(double a, double b){
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

double log_beta_function(double a, double b)
{
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

double log_beta_function(std::vector<double> alpha)
{
    double result = 0;
    unsigned long dim = alpha.size();
    for (int i = 0 ; i != dim ; ++i) {
        result += lgamma(alpha[i]);
    }
    result -= lgamma(sum(alpha));
    return result;
}

vector<double> Sample_Uniform_Distribution(boost::mt19937 &generator, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define a uniform real number distribution of integer values between 0 and 1.
    
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(0, 1));
    
    // If you want to use an STL iterator interface, use iterator_adaptors.hpp.
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}

double Sample_Uniform_Distribution(boost::mt19937 &generator)
{
    // Define a uniform real number distribution of values between 0 and 1 and sample
    boost::uniform_real<> distribution(0,1);
    return distribution(generator);
}

vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, int min, int max, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define a uniform distribution
    
    typedef boost::random::uniform_int_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(min, max));
    
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    
    return res;
}

vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, const vector<double> &probabilities, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define discrete distribution
    typedef boost::random::discrete_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(probabilities));
    boost::generator_iterator<gen_type> sample(&gen);
    
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}

int Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities)
{
    // Define discrete distribution
    boost::random::discrete_distribution<>  distribution(probabilities);
    return distribution(generator);
}

int Sample_Discrete_Distribution(boost::mt19937 &generator, double* proba_pointer, int dimension)
{
    // Define discrete distribution
    boost::random::discrete_distribution<>  distribution(proba_pointer, proba_pointer + dimension);
    return distribution(generator);
}

std::vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, double* proba_pointer, int dimension, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define discrete distribution
    typedef boost::random::discrete_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(proba_pointer, proba_pointer + dimension));
    boost::generator_iterator<gen_type> sample(&gen);
    
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}


vector<double> Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define a beta distribution
    
    typedef boost::random::beta_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(a, b));
    
    // If you want to use an STL iterator interface, use iterator_adaptors.hpp.
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
        //    cout << *it << " ";
    }
    return res;
}

double Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b)
{
    // Define beta distribution
    boost::random::beta_distribution<> distribution(a,b);
    return distribution(generator);
}

vector<double> Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define normal distribution
    typedef boost::normal_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(mu, std));
    
    // Sample
    boost::generator_iterator<gen_type> sample(&gen);
    for (it = res.begin(); it != res.end(); ++it) {
        *it = *sample++;
    }
    return res;
}

double Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std)
{
    // Define normal distribution
    boost::normal_distribution<> distribution(mu, std);
    return distribution(generator);
}

double Sample_Truncated_Normal_Distribution(boost::mt19937 &generator, double mu, double std, double min)
{
    // Define normal distribution
    boost::normal_distribution<> distribution(mu, std);
    double answer;
    do
    {
        answer = distribution(generator);
    }
    while(answer < min);
    return answer;
}

double Sample_Truncated_Normal_Distribution(boost::mt19937 &generator, double mu, double std, double min, double max)
{
    // Define normal distribution
    boost::normal_distribution<> distribution(mu, std);
    double answer;
    do
    {
        answer = distribution(generator);
    }
    while(answer < min || answer > max);
    return answer;
}


double Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta)
{
    // Define gamma distribution
    boost::gamma_distribution<> distribution(k, theta);
    return distribution(generator);
}

vector<double> Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define gamma distribution
    typedef boost::gamma_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(k, theta));
    
    // Sample
    boost::generator_iterator<gen_type> sample(&gen);
    for (it = res.begin(); it != res.end(); ++it) {
        *it = *sample++;
    }
    return res;
}


std::vector<double> Sample_Dirichlet_Distribution(boost::mt19937 &generator, double* dirichletParam, int dim)
{
    vector<double> res(dim);
    
    for (int i = 0; i < dim; ++i) {
        res[i] = Sample_Gamma_Distribution(generator, *(dirichletParam + i), 1.);
    }
    res = res/sum(res);
    return res;
}

double log_beta_pdf(double x, double a, double b)
{
    double res = lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1)*log(x) + (b - 1)*log(1 - x);
    return res;
}

double log_dirichlet_pdf(double* sample, double* dirichletParam, int K)
{
    double res = lgamma(sum(dirichletParam, K));
    for (int i = 0; i < K; ++i) {
        res += -lgamma(dirichletParam[i]) + (dirichletParam[i] - 1) * log(sample[i]);
    }
    return res;
}

vector<bool> adapted_logical_xor(vector<bool> const& states, bool const& rew)
{
    unsigned long dim = states.size();
    vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = (!states[i] != rew);
    }
    return res;
}

vector<int> stratified_resampling(boost::mt19937 &generator, vector<double> &weights)
{
    unsigned long dim = weights.size();
    vector<int> res(dim);
    vector<double> cumSum(dim);
    partial_sum(weights.begin(), weights.end(), cumSum.begin());
    cumSum               = cumSum * (double)dim;
    double uniformSample = Sample_Uniform_Distribution(generator);
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum[index] < uniformSample) { ++index;}
        res[i] = index; ++uniformSample;
    }
    return res;
}

vector<int> stratified_resampling(boost::mt19937 &generator, double weights[], int dim)
{
    vector<int> res(dim);
    vector<double> cumSum(dim);
    partial_sum(&weights[0], &weights[0] + dim, cumSum.begin());
    cumSum               = cumSum * (double)dim;
    double uniformSample = Sample_Uniform_Distribution(generator);
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum[index] < uniformSample) { ++index;}
        res[i] = index; ++uniformSample;
    }
    return res;
}

void stratified_resampling(boost::mt19937 &generator, double weights[], int dim, int* ancestors_idx)
{
    vector<int> res(dim);
    vector<double> cumSum(dim);
    partial_sum(&weights[0], &weights[0] + dim, cumSum.begin());
    cumSum               = cumSum * (double)dim;
    double uniformSample = Sample_Uniform_Distribution(generator);
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum[index] < uniformSample) { ++index;}
        *(ancestors_idx + i) = index; ++uniformSample;
    }
    return ;
}



