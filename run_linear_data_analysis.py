from pymc3 import *
from tqdm import tqdm

import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
import random

import simulate_data.simulate_data as sd


class linear_model_analysis():
    def __init__(self, data, true_beta_propensity_score, true_beta_response, predictors):
        
        self.data = data
        self.true_beta_propensity_score = true_beta_propensity_score
        self.true_beta_response = true_beta_response
        self.predictors = predictors
        self.mu_1, self.mu_0, self.Var_1, self.Var_0 = self.estimate_Y_1_0_mu_std(self.data, self.predictors)
        
    
    def model_traceplot(self, trace):
        traceplot(trace)

    def get_posterior_predictive(self, model_in, trace_in, var_names_in):
        with model_in:
            ppc = sample_posterior_predictive(
                trace_in, var_names=var_names_in, progressbar=False
            )
        return ppc
    
    def get_posterior_mean(self, model_in, trace_in, predictors, adjust_variance=False):
        
        if adjust_variance == True:
            X=self.data[predictors]
            posterior_mean = np.matmul(X.to_numpy(), trace_in["Beta"].T)
            return posterior_mean.T * trace_in["Y_i_star_std"]
        else:
            X=self.data[predictors]
            posterior_mean = np.matmul(X.to_numpy(), trace_in["Beta"].T)
            return posterior_mean.T
    
    def fit_propensity_score_model(self, niter, tune, progressbar_bool=False, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
        X=self.data[predictors].to_numpy()
        T=self.data['T'].to_numpy()
        prior_means = 0
        
        with Model() as self.propensity_score_model:
            Beta = Normal('Beta', mu=prior_means, sigma=3, shape=len(predictors))
            model_log_odds = math.dot(X, Beta)
            propensity_score = Deterministic("propensity_score", sd.inv_log_odds(model_log_odds))
            y = Bernoulli('y', p=propensity_score, observed=T)
        
        with self.propensity_score_model:
            start = find_MAP(progressbar=False)
            self.propensity_score_trace = sample(niter, tune=tune, start=start, progressbar=progressbar_bool)

    def fit_linear_model(self, niter, tune, X,Y, progressbar_bool = True, predictors=None):
        
        if not progressbar_bool:
            logger = logging.getLogger('pymc3')
            logger.setLevel(logging.ERROR)
        if predictors is None:
            predictors = self.predictors
            
        with Model() as lm_model:
            Beta = Normal('Beta', mu=0, sigma=10, shape=len(predictors))
            mu = math.dot(X, Beta)
            sig = HalfNormal('sig', sigma=5)
            expected_value_Y = Normal("expected_value_Y", mu=mu, sigma=sig, observed=Y)
        with lm_model:
            start = find_MAP(progressbar=False)
            trace = sample(niter, tune=tune, start=start, progressbar=False)
        return lm_model, trace
    
    def fit_Y_i_star_mean_propensity_linear_model(self, niter, tune, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        p = self.propensity_score_trace['propensity_score'].mean(axis=0)        
        Y=sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
        X=self.data[predictors].to_numpy()
        
        lm_model, self.Y_i_star_mean_propensity_linear_model_trace = self.fit_linear_model(
            niter, tune, X,Y,progressbar_bool = True, predictors=predictors)
        
        return lm_model, self.Y_i_star_mean_propensity_linear_model_trace
    
    def fit_Y_i_star_tilda_mean_propensity_linear_model(self, niter, tune, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        p = self.propensity_score_trace['propensity_score'].mean(axis=0)
        Y_i_star_var = sd.get_variance_Y_i_star(self.Var_1, self.Var_0, p, self.mu_1, self.mu_0)
        Y_i_star = sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
        Y = sd.get_Y_i_star_tilda(Y_i_star, np.sqrt(Y_i_star_var))
        
        X=self.data[predictors].to_numpy()
        
        lm_model, self.Y_i_star_tilda_mean_propensity_linear_model_trace_temp = self.fit_linear_model(
            niter, tune, X,Y,progressbar_bool = True, predictors=predictors)
        
        self.Y_i_star_tilda_mean_propensity_linear_model_trace={}
        self.Y_i_star_tilda_mean_propensity_linear_model_trace['Beta']=(
            self.Y_i_star_tilda_mean_propensity_linear_model_trace_temp['Beta'])
        self.Y_i_star_tilda_mean_propensity_linear_model_trace['Y_i_star_std']=np.sqrt(Y_i_star_var)
        
        return lm_model, self.Y_i_star_tilda_mean_propensity_linear_model_trace
    
    def fit_Y_i_star_true_propensity_linear_model(self, niter, tune, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        p = self.data["P(T=1)"] #self.true_propensity_score_trace['propensity_score'].mean(axis=0)
        Y=sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
        X=self.data[predictors].to_numpy()
        
        lm_model, self.Y_i_star_true_propensity_linear_model_trace = self.fit_linear_model(
            niter, tune, X,Y,progressbar_bool = True, predictors=predictors)
        
        return lm_model, self.Y_i_star_true_propensity_linear_model_trace

    def fit_Y_i_star_tilda_true_propensity_linear_model(self, niter, tune, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        p = self.data["P(T=1)"]
        Y_i_star_var = sd.get_variance_Y_i_star(self.Var_1, self.Var_0, p, self.mu_1, self.mu_0)
        Y_i_star = sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
        Y = sd.get_Y_i_star_tilda(Y_i_star, np.sqrt(Y_i_star_var))
        X=self.data[predictors].to_numpy()
        
        lm_model, self.Y_i_star_tilda_true_propensity_linear_model_trace_temp = self.fit_linear_model(
            niter, tune, X,Y,progressbar_bool = True, predictors=predictors)
        
        self.Y_i_star_tilda_true_propensity_linear_model_trace={}
        self.Y_i_star_tilda_true_propensity_linear_model_trace['Beta']=(
            self.Y_i_star_tilda_true_propensity_linear_model_trace_temp['Beta'])
        self.Y_i_star_tilda_true_propensity_linear_model_trace['Y_i_star_std']=np.sqrt(Y_i_star_var)
        
        return lm_model, self.Y_i_star_tilda_true_propensity_linear_model_trace
    
    def fit_Y_i_star_posterior_based_model(self, niter, tune, n_post_samples=100, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        X=self.data[predictors].to_numpy()
        n_post_samples = min(n_post_samples, self.propensity_score_trace['propensity_score'].shape[0])
        row_index = random.sample(range(self.propensity_score_trace['propensity_score'].shape[0]),n_post_samples)
        for j in tqdm(range(len(row_index))):
            logger = logging.getLogger('pymc3')
            logger.setLevel(logging.ERROR)
            p=self.propensity_score_trace['propensity_score'][row_index[j],:]
            Y = sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
            
            model, post_p_CATE_trace = self.fit_linear_model(
                niter, tune, X,Y,progressbar_bool = False, predictors=predictors)
            
            df_temp = pd.DataFrame(post_p_CATE_trace['Beta'], columns=range(X.shape[1]))
            if j==0:
                all_runs_CATE_trace = df_temp
            else: 
                all_runs_CATE_trace = pd.concat([all_runs_CATE_trace, df_temp], ignore_index=True)
                
        self.Y_i_star_posterior_propensity_linear_model_trace=all_runs_CATE_trace
        
    def fit_Y_i_star_tilda_posterior_based_model(self, niter, tune, n_post_samples=100, predictors=None):
        
        if predictors is None:
            predictors = self.predictors
            
        X=self.data[predictors].to_numpy()
        n_post_samples = min(n_post_samples, self.propensity_score_trace['propensity_score'].shape[0])
        row_index = random.sample(range(self.propensity_score_trace['propensity_score'].shape[0]),n_post_samples)
        for j in tqdm(range(len(row_index))):
            logger = logging.getLogger('pymc3')
            logger.setLevel(logging.ERROR)
            p=self.propensity_score_trace['propensity_score'][row_index[j],:]

            Y_i_star_var = sd.get_variance_Y_i_star(self.Var_1, self.Var_0, p, self.mu_1, self.mu_0)
            Y_i_star = sd.get_Y_i_star(self.data["Y_obs"], self.data["T"], p)
            Y = sd.get_Y_i_star_tilda(Y_i_star, np.sqrt(Y_i_star_var))       
            X=self.data[predictors].to_numpy()
            
            
            model, post_p_CATE_trace = self.fit_linear_model(
                niter, tune, X,Y,progressbar_bool = False, predictors=predictors)
            
            #if j==0:
            #    all_runs_CATE_trace = pd.DataFrame(post_p_CATE_trace['Beta'])
            #else: 
            #    all_runs_CATE_trace = pd.concat([all_runs_CATE_trace, pd.DataFrame(post_p_CATE_trace['Beta'])])
            
            X_Beta = np.matmul(X, post_p_CATE_trace['Beta'].T)
            print("X_Beta shape=",X_Beta.shape)
            print("np.sqrt(Y_i_star_var) shape=",np.sqrt(Y_i_star_var).shape)
            
            df_temp = pd.DataFrame(post_p_CATE_trace['Beta'], columns=range(X.shape[1]))
            if j==0:
                all_runs_CATE_trace = df_temp
                all_runs_variance_Y_i_star = [np.sqrt(Y_i_star_var)]
            else: 
                all_runs_CATE_trace = pd.concat([all_runs_CATE_trace, df_temp], ignore_index=True)
                all_runs_variance_Y_i_star.append(np.sqrt(Y_i_star_var))
                
        self.Y_i_star_tilda_posterior_propensity_linear_model_trace={}        
        self.Y_i_star_tilda_posterior_propensity_linear_model_trace['Beta']=all_runs_CATE_trace
        self.Y_i_star_tilda_posterior_propensity_linear_model_trace['Y_i_star_std']=all_runs_variance_Y_i_star

    

    def estimate_Y_1_0_mu_std(self, linear_data, predictors):
    
        Y_1 = np.array(linear_data.loc[linear_data['T']==1,'Y_obs'])
        X_1 = np.array(linear_data.loc[linear_data['T']==1,predictors])
    
        Y_0 = np.array(linear_data.loc[linear_data['T']==0,'Y_obs'])
        X_0 = np.array(linear_data.loc[linear_data['T']==0,predictors])
    
        mod_1 = sm.OLS(Y_1, X_1)
        res_1 = mod_1.fit()
        mu_1 = res_1.fittedvalues
        Var_1 = (np.sum((mu_1 - Y_1)*(mu_1 - Y_1)))
    
        mod_0 = sm.OLS(Y_0, X_0)
        res_0 = mod_0.fit()
        mu_0 = res_0.fittedvalues
        Var_0 = (np.sum((mu_0 - Y_0)*(mu_0 - Y_0)))
    
        return res_1.predict(linear_data[predictors]), res_0.predict(linear_data[predictors]), Var_1, Var_0


    
    
    