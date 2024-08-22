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

N = 40
#P = 10000
P = 1000
#reps = 5

exec(open('python/bernoulli.py').read())

sigma = 1.

#np.random.seed(4)
np.random.seed(123)
X = np.random.normal(size=[N,P])
beta_true = np.array([-1.08] + [0 for _ in range(P-1)])
y = X@beta_true + np.random.normal(size=N, scale = sigma) + 50
XX = np.random.normal(size=[N,P])

# Novar
#sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = True, cost_checks = False, verbose = True, sigma2_fixed = None)
sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = True, novar = True, cost_checks = False, verbose = True, sigma2_fixed = None, max_nnz = np.inf)
sbl_betas = sbl_betas.mean()
print('novar')
print(np.sum(np.square(sbl_betas-beta_true)))

# Variational.
sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = False, verbose = True, sigma2_fixed = None, max_nnz = np.inf)
sbl_betas = sbl_betas.mean()
print('var')
print(np.sum(np.square(sbl_betas-beta_true)))

beta_ncv, yy_ncv = pred_ncv(X, y, XX, lik = 'gaussian')
beta_ncv = beta_ncv[1:]
print('ncv')
print(np.sum(np.square(beta_ncv-beta_true)))
