#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  neo_ncv_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.03.2024

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import minimize_scalar
from python.tfp_plus import tri_quant
from scipy.optimize import minimize_scalar
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv

N = 100
NN = 1000
P = 10

np.random.seed(123) # Q: how come we have so much less regularization than them?
#np.random.seed(1234) # We do much better than them.
#np.random.seed(1237) # Q: Our estimate is big: bigger than accurate, bigger than ncv_reg, and even bigger than a straight-up logistic regression!
#np.random.seed(1241) 

sigma = 1.

lik = 'bernoulli'

## Compare on binomial data.
X = np.random.normal(size=[N,P])
XX = np.random.normal(size=[NN,P])
beta_true = np.array([1.] + [0 for _ in range(P-1)])
if lik=='gaussian':
    y = X@beta_true + np.random.normal(scale=sigma2_true,size=N)
    yy = XX@beta_true + np.random.normal(scale=sigma2_true,size=NN)
elif lik=='bernoulli':
    mu_y = X@beta_true 
    mu_yy = XX@beta_true 
    p_y = jax.nn.sigmoid(mu_y)
    p_yy = jax.nn.sigmoid(mu_yy)
    y = np.random.binomial(1,p=p_y)
    yy = np.random.binomial(1,p=p_yy)
else:
    raise Exception("Bad lik")

exec(open('python/bernoulli.py').read())
np.random.seed(123)
fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = True, lik = lik)
print(fst_Q.mean())

beta_ncv, yy_ncv = pred_ncv(X, y, XX, lik =lik)
print(beta_ncv)

import statsmodels.api as sm
sm.Logit(np.array(y), np.array(X)).fit().summary()
sm.Logit(np.array(y), np.array(X)[:,:1]).fit().summary()
