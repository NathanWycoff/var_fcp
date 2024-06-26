#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

resdf = pd.read_csv('sim_out/all_bake.csv')

Ns = sorted(list(set(resdf['N'])))
Ps = sorted(list(set(resdf['P'])))
pnzs = sorted(list(set(resdf['pnz'])))
reps = len(set(resdf['rep']))

nfigs = len(Ps)*len(pnzs)
ncols = 2
nrows = int(np.ceil(nfigs/ncols))

fig = plt.figure(figsize=[3*ncols,3*nrows])
for pnzi,pnz in enumerate(pnzs):
    for Pi,P in enumerate(Ps):
        plt.subplot(nrows,ncols,pnzi+Pi*len(pnzs)+1)
        dfi = resdf.loc[np.logical_and(resdf['P']==P,resdf['pnz']==pnz),:]
        dfi = dfi.drop(['P','pnz'], axis = 1)
        ub = dfi.groupby('N').quantile(0.9)
        med = dfi.groupby('N').median()
        mean = dfi.groupby('N').mean()
        lb = dfi.groupby('N').quantile(0.1)

        plt.plot(med['sbl_mse'], label = 'SBL', color = 'C0')
        plt.plot(lb['sbl_mse'], color = 'C0', linestyle = '--')
        plt.plot(ub['sbl_mse'], color = 'C0', linestyle = '--')
        plt.plot(med['ncv_mse'], label = 'NCV', color = 'C1')
        plt.plot(lb['ncv_mse'], color = 'C1', linestyle = '--')
        plt.plot(ub['ncv_mse'], color = 'C1', linestyle = '--')
        if pnzi == 0 and Pi == 0:
            plt.legend()
        plt.yscale('log')
        ax = plt.gca()
        ax1 = ax.twinx()
        ax1.plot(mean['sbl_cov'], color = 'green', linestyle = '--')
        print("Add 95 line.")

        plt.title("P="+str(P)+" pnz="+str(pnz))
plt.tight_layout()
plt.savefig("all_bake.pdf")
plt.close()
