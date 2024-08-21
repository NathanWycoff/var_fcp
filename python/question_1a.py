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
#P = 40
P = 2
reps = 5
seeds = np.arange(reps)

exec(open('python/bernoulli.py').read())

Na = 100
amin = 1.
amax = 5.
a_range = np.linspace(amin,amax,num=Na)
errs = np.nan*np.zeros([Na,Na,reps])

for si, seed in enumerate(tqdm(seeds)):
    np.random.seed(seed)
    X = np.random.normal(size=[N,P])
    beta_true = np.array([-1.08] + [0 for _ in range(P-1)])
    y = X@beta_true + np.random.normal(size=N) + 50
    XX = np.random.normal(size=[N,P])

    for i, ai in enumerate(tqdm(a_range, leave = False)):
        for j, aj in enumerate(tqdm(a_range, leave = False)):
            sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = True, cost_checks = False, A_MCP = np.array([ai, aj]), verbose = False)
            sbl_betas = sbl_betas.mean()
            #errs[i,j,si] = np.mean(np.square(sbl_betas - beta_true[np.newaxis,:]))
            errs[i,j,si] = np.min(np.mean(np.square(sbl_betas - beta_true[np.newaxis,:]), axis = 1))

err_mu = np.mean(errs, axis = 2)

fig = plt.figure()
plt.imshow(err_mu, extent = (amin, amax, amin, amax), origin = 'lower')
plt.colorbar()
plt.savefig("q1a.pdf")
plt.close()
