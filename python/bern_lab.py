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

N = 12000
NN = 1000
P = 2

np.random.seed(123)

## Compare on binomial data.
X = np.random.normal(size=[N,P])
XX = np.random.normal(size=[NN,P])
mu_y = X[:,0]
mu_yy = XX[:,0]
p_y = jax.nn.sigmoid(mu_y)
p_yy = jax.nn.sigmoid(mu_yy)
y = np.random.binomial(1,p=p_y)
yy = np.random.binomial(1,p=p_yy)

exec(open('python/bernoulli.py').read())
#np.random.seed(123)
fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = False, lik = 'bernoulli')
print(fst_Q.mean())

# TODO: Intercept
print("Hey there's no intercept :)")

beta_ncv, yy_ncv = pred_ncv(X, y, XX, lik = 'bernoulli')
beta_ncv

### Compare on normal data.
#X = np.random.normal(size=[N,P])
#XX = np.random.normal(size=[NN,P])
#mu_y = X[:,0]
#mu_yy = XX[:,0]
#y = mu_y + np.random.normal(size=N)
#yy = mu_yy + np.random.normal(size=NN)
#
#exec(open('python/bernoulli.py').read())
##np.random.seed(123)
#fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = False, lik = 'gaussian')
#print(fst_Q.mean())

### Manual lmao
#ytild = y-0.5
#np.linalg.lstsq(X/2, ytild)[0]

import statsmodels.api as sm
sm.Logit(y, X).fit().summary()

exec(open('python/theirs.py').read())
mod = VBLogisticRegression(n_iter = 100)
mod.fit(X, y)
mod.coef_

a_me = lambda xi: 0.5/xi*(jax.nn.sigmoid(xi)-0.5)
