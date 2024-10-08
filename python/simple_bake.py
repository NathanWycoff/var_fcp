#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  simple_bake.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

import numpy as np
#from python.rec_with_ncvreg_lab.py import pred_sbl
#exec(open('python/fast_boi.py').read())
exec(open('python/bernoulli.py').read())
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ranksums as rs

## Still to do:
## Intercepts!

#np.random.seed(123)
np.random.seed(271)

#lik = 'bernoulli'
lik = 'gaussian'

N = 40
P = 10000
nnz = 1

#sigma2_true = np.square(1.)
#sigma2_true = np.square(2.)
#sigma2_true = np.square(0.5)
sigma2_true = np.square(0.1)

print("hey there's no intercept.")

#N = 100
#P = 1000
#nnz = 5

#N = 10000
#P = 1000

NN = 1000

#nnz = 10
#reps = 5
#reps = 100
#reps = 30
reps = 50
#reps = 4

big_nz = False
fix_nz = True
assert not big_nz and fix_nz

penalty = 'MCP'

err_sbl = np.zeros(reps)*np.nan
cov_sbl = np.zeros(reps)*np.nan
err_ncv = np.zeros(reps)*np.nan
for rep in tqdm(range(reps)):
    #beta_nz = np.random.normal(size=nnz)
    beta_nz = np.random.normal(size=nnz)
    if big_nz:
        beta_nz = np.sign(beta_nz)*(np.abs(beta_nz)+1.)
    elif fix_nz:
        beta_nz = np.array([-1.08])
        assert nnz==1
    nz_locs = np.random.choice(P,nnz,replace=False)
    beta_true = np.zeros(P)
    beta_true[nz_locs] = beta_nz
    #beta_true = np.concatenate([[50.], beta_true])
    beta_true = np.concatenate([[0.], beta_true])

    X = np.random.normal(size=[N,P])
    X1 = np.concatenate([np.ones([N,1]), X], axis = 1)
    XX = np.random.normal(size=[NN,P])
    XX1 = np.concatenate([np.ones([NN,1]), XX], axis = 1)

    if lik=='gaussian':
        y = X1@beta_true + np.random.normal(scale=sigma2_true,size=N)
        yy = XX1@beta_true + np.random.normal(scale=sigma2_true,size=NN)
    elif lik=='bernoulli':
        mu_y = X1@beta_true 
        mu_yy = XX1@beta_true 
        p_y = jax.nn.sigmoid(mu_y)
        p_yy = jax.nn.sigmoid(mu_yy)
        y = np.random.binomial(1,p=p_y)
        yy = np.random.binomial(1,p=p_yy)
    else:
        raise Exception("Bad lik")

    #beta_sbl, yy_sbl = pred_sbl(X, y, XX, do_cv = False, doplot = True, novar = False, penalty = 'MCP', cost_checks = False, verbose = False)
    Q, yy_sbl = pred_sbl(X, y, XX, do_cv = True, doplot = True, novar = False, penalty = penalty, cost_checks = False, verbose = False, lik = lik)
    if penalty=='MCP':
        lb = tri_quant(Q, 0.025)
        ub = tri_quant(Q, 0.975)
        med = tri_quant(Q, 0.5)
    else:
        lb = Q.quantile(0.025)
        ub = Q.quantile(0.975)
        med = Q.quantile(0.5)
    beta_sbl = Q.mean()
    beta_ncv, yy_ncv = pred_ncv(X, y, XX, lik = lik)
    beta_ncv = beta_ncv[1:]

    assert beta_sbl.shape==beta_true[1:].shape
    assert beta_ncv.shape==beta_true[1:].shape
    err_sbl[rep] = np.sum(np.square(beta_sbl - beta_true[1:]))
    err_ncv[rep] = np.sum(np.square(beta_ncv - beta_true[1:]))

    covered = np.logical_and(lb[nz_locs] <= beta_true[1:][nz_locs], ub[nz_locs] >= beta_true[1:][nz_locs])
    cov_sbl[rep] = np.mean(covered)

fig = plt.figure()
#trans = np.log10
trans = lambda x: np.log10(x[~np.isnan(x)])
plt.boxplot([trans(err_ncv), trans(err_sbl)])
#plt.boxplot(err_ncv-err_sbl)
#plt.boxplot(np.log10(err_ncv/err_sbl))
plt.savefig("bake.pdf")
plt.close()

print(err_ncv)
print(err_sbl)

print(rs(err_ncv, err_sbl))
print(np.nansum(err_ncv > err_sbl) / np.sum(~np.isnan(err_sbl)))
print(np.nansum(err_ncv == err_sbl) / np.sum(~np.isnan(err_sbl)))
print(np.nansum(err_ncv < err_sbl) / np.sum(~np.isnan(err_sbl)))

