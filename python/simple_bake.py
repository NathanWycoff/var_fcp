#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  simple_bake.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

import numpy as np
#from python.rec_with_ncvreg_lab.py import pred_sbl
exec(open('python/fast_boi.py').read())
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt

## Still to do:
## Intercepts!

np.random.seed(123)

## Smol but N>P
N = 10
P = 2
nnz = 2

### Smolish and N=P
#N = 40
#P = 40
#nnz = 5


#N = 1000
#P = 100
#nnz = 50

#N = 100
#P = 1000
#nnz = 5

#N = 10000
#P = 1000

NN = 1000

#nnz = 10
reps = 30
#reps = 4

penalty = 'MCP'

err_sbl = np.zeros(reps)*np.nan
cov_sbl = np.zeros(reps)*np.nan
err_ncv = np.zeros(reps)*np.nan
for rep in tqdm(range(reps)):
    beta_nz = np.random.normal(size=nnz)
    nz_locs = np.random.choice(P,nnz,replace=False)
    beta_true = np.zeros(P)
    beta_true[nz_locs] = beta_nz
    beta_true = np.concatenate([[50.], beta_true])
    #beta_true = np.concatenate([[0.], beta_true])

    X = np.random.normal(size=[N,P])
    X1 = np.concatenate([np.ones([N,1]), X], axis = 1)
    XX = np.random.normal(size=[NN,P])
    XX1 = np.concatenate([np.ones([NN,1]), XX], axis = 1)
    sigma2_true = np.square(1)
    #sigma2_true = np.square(1e-4)
    #sigma2_true = np.square(1e4)

    y = X1@beta_true + np.random.normal(scale=sigma2_true,size=N)
    yy = XX1@beta_true + np.random.normal(scale=sigma2_true,size=NN)

    #beta_sbl, yy_sbl = pred_sbl(X, y, XX, do_cv = False, doplot = True, novar = False, penalty = 'MCP', cost_checks = False, verbose = False)
    Q, yy_sbl = pred_sbl(X, y, XX, do_cv = True, doplot = True, novar = False, penalty = penalty, cost_checks = False, verbose = False)
    if penalty=='MCP':
        lb = tri_quant(Q, 0.025)
        ub = tri_quant(Q, 0.975)
        med = tri_quant(Q, 0.5)
    else:
        lb = Q.quantile(0.025)
        ub = Q.quantile(0.975)
        med = Q.quantile(0.5)
    beta_sbl = Q.mean()
    beta_ncv, yy_ncv = pred_ncv(X, y, XX)
    beta_ncv = beta_ncv[1:]

    assert beta_sbl.shape==beta_true[1:].shape
    assert beta_ncv.shape==beta_true[1:].shape
    err_sbl[rep] = np.mean(np.square(beta_sbl - beta_true[1:]))
    err_ncv[rep] = np.mean(np.square(beta_ncv - beta_true[1:]))

    covered = np.logical_and(lb[nz_locs] <= beta_true[1:][nz_locs], ub[nz_locs] >= beta_true[1:][nz_locs])
    cov_sbl[rep] = np.mean(covered)

fig = plt.figure()
trans = np.log10
plt.boxplot([trans(err_ncv), trans(err_sbl)])
#plt.boxplot(err_ncv-err_sbl)
plt.savefig("bake.pdf")
plt.close()

print(err_ncv)
print(err_sbl)
