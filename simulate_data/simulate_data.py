import numpy as np
import pandas as pd
from scipy.stats import norm


def inv_log_odds(LO):
    """
    args: LO , float or list of floats: representing Log Odds
    returns inverse logit of LO
    """
    return 1/(1+ (np.exp(np.array(-LO))))


def get_Y_obs(Y0_given_X, Y1_given_X, W_i):
    
    Y0_given_X = np.array(Y0_given_X)
    Y1_given_X = np.array(Y1_given_X)
    W_i = np.array(W_i)
    
    Y_obs = Y0_given_X
    Y_obs[W_i == 1] = Y1_given_X[W_i == 1]
    
    return Y_obs


def get_Y_i_star(Y_obs, W_i, p):
    Y_obs = np.array(Y_obs)
    W_i = np.array(W_i)
    p = np.array(p)
    return Y_obs * (W_i - p)/(p*(1-p))


def get_variance_Y_i_star(var_1, var_0,p,mu_1,mu_0):
    var_1=np.array(var_1)
    var_0=np.array(var_0)
    p=np.array(p)
    mu_1=np.array(mu_1)
    mu_0=np.array(mu_0)
    
    odds=p/(1-p)
    output = (
        var_1/(p*(1-p)) +
        (var_0 - var_1)/(1-p) + 
        odds*mu_0*mu_0 +
        (1/odds)*mu_1*mu_1 + 
        2*mu_0*mu_1
    )
    return output


def get_Y_i_star_tilda(Y_i_star, std_Y_i_star):
    return np.array(Y_i_star)/np.array(std_Y_i_star)


def make_basic_linear_data(p, N, y_0_1_noise_scale=0.0001, log_odds_noise_scale=0.0001, random_seed = 58):

    """
    returns: basic_linear_data, lm_true_beta_propensity_scores, lm_true_beta_response
    """
    np.random.seed(seed=random_seed)

    basic_linear_data = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N,p-1)),
                         columns=list("V."+pd.Series(list(range(1,p))).apply(str))
                        )
    basic_linear_data["V.0"] = 1
    predictor_columns = list("V."+pd.Series(list(range(0,p))).apply(str))

    basic_linear_data = basic_linear_data[predictor_columns]

    lm_true_beta_propensity_scores = np.random.uniform(low=-1, high=1, size=p)
    lm_true_beta_response = np.random.uniform(low=-1, high=1, size=p)
    lm_true_beta_treatment = np.random.uniform(low=-1, high=1, size=p)
    
    mu_0 = np.matmul(basic_linear_data[predictor_columns].to_numpy(), lm_true_beta_response) 
    basic_linear_data["Y0_given_X"] = (
        mu_0 + np.random.normal(loc=0, scale=y_0_1_noise_scale, size=basic_linear_data.shape[0])
    )
    mu_1 = mu_0 + np.matmul(basic_linear_data[predictor_columns].to_numpy(), lm_true_beta_treatment)
    basic_linear_data["Y1_given_X"]  = (
        basic_linear_data["Y0_given_X"] + np.matmul(basic_linear_data[predictor_columns].to_numpy(), lm_true_beta_treatment)
        #np.random.normal(loc=treatment_effect, scale=treatment_effect_noise, size = basic_linear_data.shape[0])
    )
    basic_linear_data["Tau"] = basic_linear_data["Y1_given_X"]-basic_linear_data["Y0_given_X"]
    
    log_odds = (
        np.matmul(basic_linear_data[predictor_columns].to_numpy(), lm_true_beta_propensity_scores) + 
        np.random.normal(loc=0, scale=log_odds_noise_scale, size=basic_linear_data.shape[0])
    )
    
    p=inv_log_odds(log_odds)
    basic_linear_data["P(T=1)"] = p
    basic_linear_data["T"] = np.random.binomial(n=1, p=basic_linear_data["P(T=1)"])
    
    
    basic_linear_data["Y_obs"] = get_Y_obs(
        basic_linear_data["Y0_given_X"], basic_linear_data["Y1_given_X"], basic_linear_data["T"])
    
    basic_linear_data["mu_0"] = mu_0
    basic_linear_data["mu_1"] = mu_1
    basic_linear_data["var_1"] = y_0_1_noise_scale*y_0_1_noise_scale
    
    var_1 = var_0 = y_0_1_noise_scale*y_0_1_noise_scale
    
    basic_linear_data["Var(Y_i_star)_true"] = get_variance_Y_i_star(var_1, var_0,p,mu_1,mu_0)
    
    basic_linear_data["Y_i_star_true"] = get_Y_i_star(basic_linear_data["Y_obs"], basic_linear_data["T"], p)
    
    basic_linear_data["Y_i_star_tilda_true"] = get_Y_i_star_tilda(
        basic_linear_data["Y_i_star_true"], np.sqrt(np.array(basic_linear_data["Var(Y_i_star)_true"])))
    
    return basic_linear_data, lm_true_beta_propensity_scores, lm_true_beta_response, predictor_columns


def make_hahn_data(function_type="linear", effect_type="heterogeneous", n_in_study=500):
# Five variables comprise x; the first three are continuous, drawn as standard normal random variables, the fourth is a dichotomous variable and the fifth is unordered categorical, taking three levels (denoted 1, 2, 3).
    def g(x):
        # g(1) = 2, g(2) = −1 and g(3) = −4
        return -2*x+5
    
    x_1 = np.random.normal(loc=0, scale=1, size=n_in_study)
    x_2 = np.random.normal(loc=0, scale=1, size=n_in_study)
    x_3 = np.random.normal(loc=0, scale=1, size=n_in_study)
    x_4 = np.random.choice([1,2,3], size=n_in_study)
    x_5 = np.random.binomial(n=1,p=.5, size=n_in_study)
    
    if effect_type == "homogeneous":
        # τ(x) = 3, homogeneous
        Tau=3
    if effect_type == "heterogeneous":
        # τ(x) = 1 + 2*x_2*x_5, heterogeneous,
        Tau = 1 + 2*x_2*x_5
    
    if function_type == "linear":
        # μ(x) = 1 + g(x4) + x1x3, linear,
        mu = 1 + g(x_4) + x_1*x_3
    if function_type == "nonlinear":
        # μ(x) = −6 + g(x4) + 6|x3 − 1|, nonlinear,
        mu = -6 + g(x_4) + 6*np.absolute(x_3 - 1)
    # the propensity function is π(xi) = 0.8*Φ(3*μ(x_i)/s − 0.5*x_1) + 0.05 + u_i/10,
    # where s is the standard deviation of μ taken over the observed sample and u_i ∼ Uniform(0, 1).
    s = np.std(mu)
    u_i=np.random.uniform(size=n_in_study)
    pi = 0.8*norm.cdf(3*mu/s - 0.5*x_1) + 0.05 + u_i/10
    w_i = np.random.binomial(n=1,p=pi, size=n_in_study)
    
    dummies = pd.get_dummies(x_4)
    dummies.columns = ["x_4_" + i for i in dummies.columns.values.astype(str)] 

    output=pd.DataFrame(
        {
            "x_0":1,
            "x_1":x_1,
            "x_2":x_2,
            "x_3":x_3,
            "x_4":x_4,
            "x_5":x_5,
            "w_i":w_i,
            "mu":mu,
            "Tau":Tau,
            "pi":pi,
            "x_1_x_3":x_1*x_3,
            "x_2_x_5":x_2*x_5,
            "x_4_1": dummies.x_4_1, "x_4_2": dummies.x_4_2, "x_4_3": dummies.x_4_3,
            "P(T=1)":pi,
            "T":w_i,
            "Y0_given_X": mu,
            "Y1_given_X": mu+Tau,
            "Y_obs":mu+Tau*w_i,
        }
    )
    return output
