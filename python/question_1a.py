#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
P = 2

np.random.seed(123) # Q: how come we have so much less regularization than them?

sigma = 1.

lik = 'gaussian'

## Compare on binomial data.
X = np.random.normal(size=[N,P])
XX = np.random.normal(size=[NN,P])
#beta_true = np.array([0.5] + [0 for _ in range(P-1)])
beta_true = np.array([0.4] + [0 for _ in range(P-1)])
if lik=='gaussian':
    y = X@beta_true + np.random.normal(scale=sigma,size=N)
    yy = XX@beta_true + np.random.normal(scale=sigma,size=NN)
elif lik=='bernoulli':
    mu_y = X@beta_true 
    mu_yy = XX@beta_true 
    p_y = jax.nn.sigmoid(mu_y)
    p_yy = jax.nn.sigmoid(mu_yy)
    y = np.random.binomial(1,p=p_y)
    yy = np.random.binomial(1,p=p_yy)
else:
    raise Exception("Bad lik")

#exec(open('python/bernoulli.py').read())
exec(open('python/valencia.py').read())
np.random.seed(123)
#fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = False, novar = True, cost_checks = False, lik = lik, penalty = 'MCP')
fst_Q, _ = pred_sbl(X, y, XX, do_cv = False, novar = True)
my_est = fst_Q.mean().squeeze()

#beta_ncv, yy_ncv = pred_ncv(X, y, XX, lik =lik)
beta_ncv, yy_ncv = pred_ncv_no_cv(X, y, XX, lik =lik)
patricks_est = beta_ncv[1:,:].T

my_est - patricks_est

