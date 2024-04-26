#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  simple_bake.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

import numpy as np
#from python.rec_with_ncvreg_lab.py import pred_sbl
exec(open('python/rec_with_ncvreg_lab.py').read())
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt

## Still to do:
## Intercepts!

N = 100
NN = 1000
P = 100
#N = 10000
#P = 1000

nnz = 10
reps = 30

err_sbl = np.zeros(reps)*np.nan
err_ncv = np.zeros(reps)*np.nan
for rep in tqdm(range(reps)):
    beta_nz = np.random.normal(size=nnz)
    nz_locs = np.random.choice(P,nnz,replace=False)
    beta_true = np.zeros(P)
    beta_true[nz_locs] = beta_nz
    #beta_true = np.concatenate([[50.], beta_true])
    #beta_true = np.concatenate([[0.], beta_true])

    X = np.random.normal(size=[N,P])
    #X1 = np.concatenate([np.ones([N,1]), X], axis = 1)
    XX = np.random.normal(size=[NN,P])
    #XX1 = np.concatenate([np.ones([NN,1]), XX], axis = 1)
    sigma2_true = np.square(1)

    y = X@beta_true + np.random.normal(scale=sigma2_true,size=N)
    yy = XX@beta_true + np.random.normal(scale=sigma2_true,size=NN)

    # CV
    beta_sbl, yy_sbl = pred_sbl(X, y, XX)
    beta_ncv, yy_ncv = pred_ncv(X, y, XX)
    beta_ncv = beta_ncv[1:]

    # Funsies
    #beta_sbl, yy_sbl = pred_sbl(X, y, XX, do_cv = False)
    beta_sbl, yy_sbl = pred_sbl(X, y, XX, do_cv = True)
    beta_ncv, yy_ncv = pred_ncv(X, y, XX)
    beta_ncv = beta_ncv[1:]

    err_sbl[rep] = np.mean(np.square(beta_sbl - beta_true))
    err_ncv[rep] = np.mean(np.square(beta_ncv - beta_true))

fig = plt.figure()
plt.boxplot([err_ncv, err_sbl])
#plt.boxplot(err_ncv-err_sbl)
plt.savefig("bake.pdf")
plt.close()
