#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  simple_bake.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

import numpy as np
from python.dcp_lib_generic import SblNet
from python.ncvreg_wrapper import pred_ncv
from tqdm import tqdm
import matplotlib.pyplot as plt

## Still to do:
## Intercepts!

N = 1000
NN = 1000
P = 1000
#N = 10000
#P = 1000

nnz = 100
reps = 30

err_sbl = np.zeros(reps)*np.nan
err_ncv = np.zeros(reps)*np.nan
for rep in tqdm(range(reps)):
    beta_nz = np.random.normal(size=nnz)
    nz_locs = np.random.choice(P,nnz,replace=False)
    beta_true = np.zeros(P)
    beta_true[nz_locs] = beta_nz
    beta_true = np.concatenate([[50.], beta_true])

    X = np.random.normal(size=[N,P])
    X1 = np.concatenate([np.ones([N,1]), X], axis = 1)
    XX = np.random.normal(size=[NN,P])
    XX1 = np.concatenate([np.ones([NN,1]), XX], axis = 1)
    sigma2_true = np.square(1)

    y = X1@beta_true + np.random.normal(scale=sigma2_true,size=N)
    yy = XX1@beta_true + np.random.normal(scale=sigma2_true,size=NN)

    #sbl = SblNet(X, y, XX=XX, penalty = 'MCP')
    sbl = SblNet(X, y, XX=XX, penalty = 'laplace')
    beta_sbl, yy_sbl = sbl.cv_fit_predict()
    beta_ncv, yy_ncv = pred_ncv(X, y, XX)

    err_sbl[rep] = np.mean(np.square(beta_sbl - beta_true))
    err_ncv[rep] = np.mean(np.square(beta_ncv - beta_true))

fig = plt.figure()
plt.boxplot([err_ncv, err_sbl])
#plt.boxplot(err_ncv-err_sbl)
plt.savefig("bake.pdf")
plt.close()

