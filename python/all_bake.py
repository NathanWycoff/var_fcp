#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
exec(open('python/valencia.py').read())
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

## Still to do:
## Intercepts!

np.random.seed(123)

NN = 1000
reps = 30
#reps = 20
penalty = 'MCP'

## Smol but N>P

#Ns = [10, 20]
#Ns = [10, 100,1000]
#Ps = [10, 100, 1000]
Ns = [10, 100, 1000]
Ps = [10, 100]
#nnzs = [1, 5, 10]
pnzs = [0.01, 0.1, 0.5]

resdf = pd.DataFrame(np.zeros([len(Ns)*len(Ps)*len(pnzs)*reps,7]))
resdf.columns = ['N','P','pnz','rep','sbl_mse','ncv_mse','sbl_cov']

out_ind = -1
for N in tqdm(Ns):
    for P in tqdm(Ps, leave = False):
        for pnz in tqdm(pnzs, leave = False):
            nnz = int(np.ceil(P*pnz))
            for rep in tqdm(range(reps), leave = False):
                out_ind += 1

                # Generate Data
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
                sbl_mse = np.mean(np.square(beta_sbl - beta_true[1:]))
                ncv_mse = np.mean(np.square(beta_ncv - beta_true[1:]))

                covered = np.logical_and(lb[nz_locs] <= beta_true[1:][nz_locs], ub[nz_locs] >= beta_true[1:][nz_locs])
                sbl_cov = np.mean(covered)

                resdf.iloc[out_ind,:] = [N,P,pnz,rep,sbl_mse, ncv_mse, sbl_cov]


resdf.to_csv('sim_out/all_bake.csv')
