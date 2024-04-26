#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  rec_with_ncvreg_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.25.2024

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import numpy as np
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv

X = np.array([[1,2],[2,4],[3,9],[4,16]]).astype(float)
XX = np.array([[7,8]]).astype(float)
y = np.array([1,2,3,4]).astype(float)

ncv_betas, ncv_preds = pred_ncv_no_cv(X, y, XX)

################################################################################################
## Us
penalty = 'MCP'; plotname = 'traj.pdf'; add_intercept = False; scale = True; verbose = False
N,P = X.shape
X = jnp.array(X)
y = jnp.array(y)
if XX is not None:
    XX = jnp.array(XX)
    NN,PP = XX.shape
    assert PP==P
else:
    XX = jnp.array(np.nan)

#if add_intercept:
#    X = np.concatenate([np.ones([N,1]), X], axis = 1)
#    if XX is not None:
#        XX = np.concatenate([np.ones([XX.shape[0],1]), XX], axis = 1)
#    P += 1

# do CV splits.
do_cv = True
#do_cv = False
if do_cv:
    #cv_folds = 5
    cv_folds = 2
    K = cv_folds+1
    inds = np.arange(N)
    np.random.shuffle(inds)
    #test_inds = np.array_split(inds, cv_folds)
    print("Oh no! These are not the real test inds!")
    test_inds = [np.array([0,1]),np.array([2,3])]

    errs = list()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for cv_i in range(cv_folds):
        testi = test_inds[cv_i]
        traini = np.setdiff1d(np.arange(N),test_inds[cv_i])

        X_train.append(X[traini,:])
        y_train.append(y[traini])
        X_test.append(X[testi,:])
        y_test.append(y[testi])

    X_train.append(X)
    y_train.append(y)
    X_test.append(XX)
    y_test.append(jnp.array(np.nan))
else:
    K = 1
    X_train = [X]
    y_train = [y]
    X_test = [XX]
    y_test = [np.nan]


#XX_train[k] = (XX_train[k] - mu_X[k][np.newaxis,:]) / sig_X[k][np.newaxis,:]

if scale:
    mu_X = []
    sig_X = []
    mu_y = []
    sig_y = []
    for k in range(K):
        eps_div = 1e-6
        mu_X.append(np.mean(X_train[k], axis = 0))
        sig_X.append(np.std(X_train[k], axis = 0)+ eps_div)
        mu_y.append(np.mean(y_train[k]))
        X_train[k] = (X_train[k] - mu_X[k][np.newaxis,:]) / sig_X[k][np.newaxis,:]
        X_test[k] = (X_test[k] - mu_X[k][np.newaxis,:]) / sig_X[k][np.newaxis,:]

        #sig_y = np.std(y) + eps_div
        sig_y.append(1.)
        y_train[k] = (y_train[k] - mu_y[k]) / sig_y[k]

## Get tau_max
MCP_LAMBDA_max = np.max(np.abs(X_train[-1].T @ y_train[-1]))/N # Evaluate range on full dataset.
MCP_LAMBDA_min = 1e-3*MCP_LAMBDA_max
T = 100
MCP_LAMBDA_range = np.flip(np.logspace(np.log10(MCP_LAMBDA_min), np.log10(MCP_LAMBDA_max), num = T))
sdy = np.std(y_train[-1])

exec(open("python/lib.py").read())
update_eta = jax.jit(update_eta_pre)
update_lam = jax.jit(update_lam_pre)

#max_iters = 10000
max_iters = 100
#block_thresh = 1e-6
block_thresh = 1e-4

sigma2_hat = 0.*jnp.array([np.mean(np.square(yk)) for yk in y_train])+1. # Isn't this folded into the tau path?
x2 = 0.*jnp.array([jnp.sum(jnp.square(Xk), axis=0) for Xk in X_train])+1. # this is just N when scaled. 
s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?

max_nnz = 40


## Init params
eta = jnp.zeros([K,P])
#lam = jnp.array(1/np.sqrt(s))

etas = np.zeros([T, K, P])*np.nan
lams = np.zeros([T, K, P])*np.nan
sigma2s = np.zeros([T,K])*np.nan

Ns = jnp.array([Xk.shape[0] for Xk in X_train])
NNs = jnp.array([Xk.shape[0] for Xk in X_test])
yy_hat = [np.zeros([T,NNs[k]])*np.nan for k in range(K)]
preds = [X_train[k] @ eta[k,:] for k in range(K)]

lam_maxit = 100

it = 0
#for t, tau in enumerate(tqdm(tau_range)):
t, MCP_LAMBDA = 0, MCP_LAMBDA_max
#t, MCP_LAMBDA = 1, MCP_LAMBDA_range[1]
for t, MCP_LAMBDA in enumerate(tqdm(MCP_LAMBDA_range)):
    #tau_effective = tau*Ns/Ns[-1]
    a = 3.
    lam = np.ones([K, P]) * 1/(a*MCP_LAMBDA)
    tau_effective = jnp.array([a*jnp.square(MCP_LAMBDA) for _ in range(K)])

    diff = np.inf
    while (it < max_iters) and (diff > block_thresh*sdy):
        it += 1
        eta_last = jnp.copy(eta)
        #lam_last = jnp.copy(lam)
        eta, preds = update_eta_pre(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds)
        print(eta)
        #lam, lam_it = update_lam(eta, lam, tau_effective, s, max_iters = lam_maxit)
        lam_it = 0

        if lam_it == lam_maxit and verbose:
            print("Reached max iters on lam update.")

        #diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])
        diff = jnp.max(jnp.abs(eta_last-eta))
        print(diff)


    etas[t,:,:] = eta
    lams[t,:,:] = lam
    for k in range(K):
        yy_hat[k][t,:] = X_test[k] @ eta[k,:]

    nnz = np.sum(eta[-1,::]!=0)
    if nnz >= max_nnz:
        break

if verbose:
    if it==max_iters:
        print("Reached Max Iters on outer block.")
    #print("Stopped main loop on tau %d after %d iters with %d nnz"%(t,it,np.sum(eta!=0)))

errs = jnp.zeros(T)
for k in range(K-1):
    errs += np.mean(np.square(yy_hat[k] - y_test[k]), axis = 1) / (K-1)
tau_opti = np.nanargmin(errs)
tau_opt = MCP_LAMBDA_range[tau_opti]

if scale:
    for k in range(K):
        yy_hat[k] = sig_y[k]*yy_hat[k] + mu_y[k]

        etas[:,k,:] = (sig_y[k] / sig_X[k])[np.newaxis,:] * etas[:,k,:]
        lams[:,k,:] = np.square(sig_X[k] / sig_y[k])[np.newaxis,:] * lams[:,k,:]

NV_plot = np.min([5,P])
k_plot = -1

print(np.max(np.abs(etas[:,-1,0]-ncv_betas[1,:])))
print(np.max(np.abs(ncv_preds - yy_hat[-1].flatten())))

Q = get_Q(etas, lams)
#return Q.mean()[tau_opti,-1,:], yy_hat[-1][tau_opti,:]
