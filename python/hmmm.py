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
P = 10000
#reps = 5

exec(open('python/bernoulli.py').read())

sigma = 1.

np.random.seed(4)
X = np.random.normal(size=[N,P])
beta_true = np.array([-1.08] + [0 for _ in range(P-1)])
y = X@beta_true + np.random.normal(size=N, scale = sigma) + 50
XX = np.random.normal(size=[N,P])

# Novar
sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = True, cost_checks = False, verbose = True, sigma2_fixed = None)
sbl_betas = sbl_betas.mean()
#errs[i,j,si] = np.mean(np.square(sbl_betas - beta_true[np.newaxis,:]))
diffs = sbl_betas - beta_true[np.newaxis,:]
sse_vs_tau = np.sum(np.square(diffs), axis = 1)
optind = np.argmin(sse_vs_tau)
sbl_betas[optind,:]
print(np.min(sse_vs_tau))

# Variational.
sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = False, cost_checks = False, verbose = True, sigma2_fixed = None)
sbl_betas = sbl_betas.mean()
#errs[i,j,si] = np.mean(np.square(sbl_betas - beta_true[np.newaxis,:]))
diffs = sbl_betas - beta_true[np.newaxis,:]
sse_vs_tau = np.sum(np.square(diffs), axis = 1)
optind = np.argmin(sse_vs_tau)
sbl_betas[optind,:]
print(np.min(sse_vs_tau))


err_mu = np.mean(errs, axis = 2)

fig = plt.figure()
plt.imshow(err_mu, extent = (amin, amax, amin, amax), origin = 'lower')
plt.colorbar()
plt.savefig("q1a.pdf")
plt.close()


i,j = np.unravel_index(np.argmin(err_mu), err_mu.shape)
a_range[i]
a_range[j]
err_mu[i,j]
err_mu[Na//2,Na//2]

