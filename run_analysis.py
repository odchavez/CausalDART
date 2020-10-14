import pandas as pd
import numpy as np
import arviz as az
import pylab as plt
from tqdm import tqdm
import random
from pymc3 import *
from scipy.stats import norm, ttest_ind

import simulate_data.simulate_data as sd
import analysis_code.run_linear_data_analysis as rlda


def run_linear_model_analysis(iterations, tune, n_post_samples, n_predictors, n_in_study, y_0_1_noise, log_odds_noise, seed):
    # Create the Data
    linear_data, true_beta_propensity_scores, true_beta_response, predictors = sd.make_basic_linear_data(
        p=n_predictors, N=n_in_study, 
        y_0_1_noise_scale=y_0_1_noise, 
        random_seed = seed
    )
    #Initialize Analysis Object
    analysis_obj = rlda.linear_model_analysis(
        linear_data, 
        true_beta_propensity_scores, 
        true_beta_response, 
        predictors
    )
    # Fit Propensity Model
    analysis_obj.fit_propensity_score_model(niter=iterations, tune=tune)
    # Get Posterior of Propensity Model
    ps_post_pred = analysis_obj.get_posterior_predictive(
        model_in=analysis_obj.propensity_score_model, 
        trace_in = analysis_obj.propensity_score_trace,
        var_names_in = ['propensity_score'])
    # Fit Y_I_star Model with Mean Propensity Score
    Y_i_star_mean_propensity_linear_model_model, Y_i_star_mean_propensity_linear_model_trace = (
        analysis_obj.fit_Y_i_star_mean_propensity_linear_model(niter=iterations, tune=tune))
    # Fit Y_I_star Model with True Propensity Score
    Y_i_star_true_propensity_linear_model_model, Y_i_star_true_propensity_linear_model_trace =(
        analysis_obj.fit_Y_i_star_true_propensity_linear_model(niter=iterations, tune=tune))
    # Fit Y_i_star Model with Posterior Based Propensity Scores
    analysis_obj.fit_Y_i_star_posterior_based_model(niter=iterations, tune=tune, n_post_samples=n_post_samples)
    
    #Make Plots
    sub_N = n_in_study-1
    X=analysis_obj.data[analysis_obj.predictors]
    posterir_fp_Tau = np.matmul(
        X.to_numpy(), 
        analysis_obj.Y_i_star_posterior_propensity_linear_model_trace.to_numpy().T)
    fp_Tau = pd.DataFrame(posterir_fp_Tau.T)
    
    plt.figure(figsize=(20,15))
    # PLOT 1
    plt.subplot(3, 2, 1)
    temp_fp = fp_Tau.loc[:, :sub_N]
    temp2 = temp_fp.reindex(temp_fp.mean(axis=0).sort_values().index, axis=1)
    _=temp2.boxplot()
    temp_Tau = analysis_obj.data.loc[:sub_N, 'Tau']
    plt.plot(range(1,sub_N+2),temp_Tau[temp2.columns], 'x', color='red')
    interval_high = np.quantile(fp_Tau, q=0.95, axis=0)
    interval_low = np.quantile(fp_Tau, q=0.05, axis=0)
    fp_max_violation = interval_high < analysis_obj.data.loc[:, 'Tau']
    fp_min_violation = interval_low > analysis_obj.data.loc[:, 'Tau']
    coverage_fp = 1-(sum(fp_max_violation)+sum(fp_min_violation))/len(fp_min_violation) 
    mil = str(round(np.mean(interval_high - interval_low),3))
    efp = fp_Tau.mean(axis=0) - analysis_obj.data.loc[:, 'Tau']
    rmse = str(round(np.sqrt(np.mean(efp*efp)), 2))
    _=plt.title("Multi Posterior - Coverage:" + str(coverage_fp) + ":::RMSE:"+ rmse + ":::MIL:" + mil)
    
    # PLOT 2
    plt.subplot(3, 2, 2)
    post_mean_Y_i_star_mean_propensity_linear_model_trace = pd.DataFrame(analysis_obj.get_posterior_mean(
        model_in=Y_i_star_mean_propensity_linear_model_model, 
        trace_in=Y_i_star_mean_propensity_linear_model_trace, 
        predictors=predictors))
    
    temp3 = post_mean_Y_i_star_mean_propensity_linear_model_trace.reindex(temp_fp.mean().sort_values().index, axis=1)
    _=temp3.boxplot()
    plt.plot(range(1,sub_N+2),temp_Tau[temp2.columns], 'x', color='red')
    interval_high = np.quantile(post_mean_Y_i_star_mean_propensity_linear_model_trace, q=0.95, axis=0)
    interval_low = np.quantile(post_mean_Y_i_star_mean_propensity_linear_model_trace, q=0.05, axis=0)
    sp_max_violation = interval_high < analysis_obj.data.loc[:, 'Tau']
    sp_min_violation = interval_low > analysis_obj.data.loc[:, 'Tau']
    coverage_sp = 1-(sum(sp_max_violation)+sum(sp_min_violation))/len(sp_min_violation)
    mil = str(round(np.mean(interval_high - interval_low),3))
    esp = post_mean_Y_i_star_mean_propensity_linear_model_trace.mean(axis=0) - analysis_obj.data.loc[:, 'Tau']
    rmse = str(round(np.sqrt(np.mean(esp*esp)), 2))    
    
    _=plt.title("Single Posterior - Coverage:" + str(coverage_sp)+ "::RMSE:"+ rmse + ":::MIL:" + mil )
    ####################################################################
    # PLOT TEST
    plt.subplot(3, 2, 3)
    post_mean_Y_i_star_true_propensity_linear_model_trace = pd.DataFrame(analysis_obj.get_posterior_mean(
        model_in=Y_i_star_true_propensity_linear_model_model, 
        trace_in=Y_i_star_true_propensity_linear_model_trace, 
        predictors=predictors))
    
    temp3 = post_mean_Y_i_star_true_propensity_linear_model_trace.reindex(temp_fp.mean().sort_values().index, axis=1)
    _=temp3.boxplot()
    plt.plot(range(1,sub_N+2),temp_Tau[temp2.columns], 'x', color='red')
    interval_high = np.quantile(post_mean_Y_i_star_true_propensity_linear_model_trace, q=0.95, axis=0)
    interval_low = np.quantile(post_mean_Y_i_star_true_propensity_linear_model_trace, q=0.05, axis=0)
    sp_max_violation = interval_high < analysis_obj.data.loc[:, 'Tau']
    sp_min_violation = interval_low > analysis_obj.data.loc[:, 'Tau']
    coverage_sp = 1-(sum(sp_max_violation)+sum(sp_min_violation))/len(sp_min_violation)
    mil = str(round(np.mean(interval_high - interval_low),3))
    esp = post_mean_Y_i_star_true_propensity_linear_model_trace.mean(axis=0) - analysis_obj.data.loc[:, 'Tau']
    rmse = str(round(np.sqrt(np.mean(esp*esp)), 2))    
    
    _=plt.title("Single Posterior - Coverage:" + str(coverage_sp)+ "::RMSE:"+ rmse + ":::MIL:" + mil )
    ####################################################################
    # PLOT 3
    plt.subplot(3, 2, 3)
    _=plt.plot(range(len(temp2.var(axis=0)[:sub_N+1])),temp2.var(axis=0)[:sub_N+1]/temp3.var(axis=0))
    _=plt.hlines((temp2.var(axis=0)/temp3.var(axis=0)).mean(), xmin=0, xmax=len(temp2.var(axis=0)[:sub_N+1]))
    _=plt.title(
        "variance ratio Multi-PPS/Single-Mean-PS="+str(round((temp2.var(axis=0)/temp3.var(axis=0)).mean(),3)))
    
    # PLOT 4
    plt.subplot(3, 2, 4)
    _=plt.hist(linear_data["Y1_given_X"], alpha=.5)
    _=plt.hist(linear_data["Y0_given_X"], alpha=.5)
    result=ttest_ind(
        linear_data["Y1_given_X"], 
        linear_data["Y0_given_X"], 
    )
    _=plt.title("Distirbutions: Y0(red) and Y1(blue) with t(Y1-Y0) pval="+str(round(result.pvalue,3)))
    
    # PLOT 5
    plt.subplot(3, 2, 5)
    plt.plot(analysis_obj.data.loc[:sub_N, 'Tau'], fp_Tau.loc[:, :sub_N].mean(axis=0), 'x', alpha=.5)
    plt.plot(analysis_obj.data.loc[:sub_N, 'Tau'],post_mean_Y_i_star_mean_propensity_linear_model_trace.mean(axis=0), 'o', alpha=.5)
    _=plt.title("E(Mul&Mean-Posterior Y_i_star) vs True Tau|X")
    
    
    # PLOT 6
    ax = plt.subplot(3, 2, 6)
    plt.plot(post_mean_Y_i_star_mean_propensity_linear_model_trace.mean(axis=0), fp_Tau.loc[:, :sub_N].mean(axis=0), 'x')
    _=plt.title("E(Muli-Posterior Y_i_star) vs E(Mean-Posterior Y_i_star)")