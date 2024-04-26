#!/usr/bin/env python3
# -*- coding: utf-7 -*-
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

#X = np.array([[1,2],[2,4],[3,9],[4,16]]).astype(float)
#XX = np.array([[7,8]]).astype(float)
#y = np.array([1,2,3,4]).astype(float)

np.random.seed(124)
N = 400
P = 40
X = np.random.normal(size=[N,P])
y = np.random.normal(size=[N])
XX = np.random.normal(size=[N,P])

ncv_betas, ncv_preds = pred_ncv_no_cv(X, y, XX)

#pred_sbl(X, y, XX)

################################################################################################
## Us
#def pred_sbl(X, y, XX = None, penalty = 'MCP', add_intercept = True, scale = True, verbose = False, do_cv = True):
penalty = 'MCP'; add_intercept = True; scale = True; verbose = False; do_cv = False
N,P = X.shape
X = jnp.array(X)
y = jnp.array(y)
if XX is not None:
    XX = jnp.array(XX)
    NN,PP = XX.shape
    assert PP==P
else:
    XX = jnp.array(np.nan)

### FCP/Variational Specification
if penalty=='laplace':
    def prox_P(x, s):
        true_pred = lambda: 0.
        false_pred = lambda: x + jnp.sign(x) * lambertw(-s * jnp.exp(-jnp.abs(x)))
        ret = jax.lax.cond(jnp.abs(x) < s, true_pred, false_pred)
        return ret

    P_FCP = lambda x: -jnp.exp(-jnp.abs(x))
    #dP_FCP = lambda x: jnp.sign(x)*jnp.exp(-jnp.abs(x))

    get_Q = lambda eta, lam: tfd.Laplace(loc=eta, scale = 1/lam)

elif penalty=='MCP':
    def prox_P(x, s):
        interp = jnp.sign(x)*(jnp.abs(x)-s)/(1.-s)
        smol_s = jnp.minimum(x,jnp.maximum(0.,interp))
        # Protect against division by 0 in case s=1.
        big_s = jax.lax.cond(jnp.abs(x)<s, lambda: 0., lambda: x)
        ret = jax.lax.cond(s<1., lambda: smol_s, lambda: big_s)
        return ret

    P_FCP = lambda x: 0.5 * jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.abs(x)-jnp.square(x), lambda: 1.)

    get_Q = lambda eta, lam: tfp.distributions.Triangular(low=eta-1/lam, high=eta+1/lam, peak=eta)
else:
    raise Exception("Unknown Penalty")

# Extract other features of penalty.
#dP_FCP = jax.vmap(jax.grad(P_FCP))
dP_FCP = jax.vmap(jax.vmap(jax.grad(P_FCP)))
v_f = get_Q(0,1).variance()

## Lambda update functions.
def body_fun_lam(val):
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = val
    new_lam = jnp.power(v_f/(s*(eta*dP_FCP(lam*eta)+1/lam)), 1./3)
    diff = jnp.max(jnp.abs(new_lam-lam))
    return eta, new_lam, tau_effective, s, diff, thresh, it+1, max_iters

def cond_fun_lam(val):
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = val
    return jnp.logical_and(diff > thresh, it<max_iters)

def update_lam_pre(eta, lam, tau_effective, s, thresh = 1e-6, max_iters = 100):
    it = 0
    thresh = 1e-6

    diff = np.inf
    val = (eta, lam, tau_effective, s, diff, thresh, 0, max_iters)
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun_lam, body_fun_lam, val)
    return lam, it

def prox_MCP_manual(z, tau, lam):
    a = 1/(jnp.square(lam)*tau)
    gamma = lam*tau

    isgtlt = (jnp.abs(z) > gamma).astype(int)
    isgtt = (jnp.abs(z)>a*gamma).astype(int)
    ind = isgtlt + isgtt
    branches = []
    branches.append(lambda: 0.)
    branches.append(lambda: jnp.sign(z)*(jnp.abs(z)-gamma)/(1-1/a))
    branches.append(lambda: z)
    ret = jax.lax.switch(ind, branches)

    return ret

## eta update functions
def body_fun_eta(p, val):
    eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = val
    for k in range(K):
        pred_other = preds[k] - eta[k,p] * X_train[k][:,p]
        resid_other = y_train[k] - pred_other
        #xdn2 = sigma2_hat[k]/s[k,p]
        #ols = jnp.sum(X_train[k][:,p] * resid_other) / xdn2
        ols = jnp.mean(X_train[k][:,p] * resid_other)

        #eta_new = prox_P(ols*lam[k,p], s[k,p]*jnp.square(lam[k,p])*tau_effective[k])/lam[k,p]
        print("Warning: manual prox")
        assert penalty=='MCP'
        eta_new = prox_MCP_manual(ols, tau_effective[k], lam[k,p])
        eta = eta.at[k,p].set(eta_new)

        preds[k] = pred_other + eta[k,p] * X_train[k][:,p]

    return eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train

def update_eta_pre(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds):
    val = (eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train)
    eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = jax.lax.fori_loop(0, P, body_fun_eta, val)

    return eta, preds
########################################

#if add_intercept:
#    X = np.concatenate([np.ones([N,1]), X], axis = 1)
#    if XX is not None:
#        XX = np.concatenate([np.ones([XX.shape[0],1]), XX], axis = 1)
#    P += 1

# do CV splits.
#do_cv = False
if do_cv:
    cv_folds = 10
    #cv_folds = 2
    K = cv_folds+1
    inds = np.arange(N)
    np.random.shuffle(inds)
    test_inds = np.array_split(inds, cv_folds)

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
        y_test[k] = (y_test[k] - mu_y[k]) / sig_y[k]

## Get tau_max
MCP_LAMBDA_max = np.max(np.abs(X_train[-1].T @ y_train[-1]))/N # Evaluate range on full dataset.
MCP_LAMBDA_min = 1e-3*MCP_LAMBDA_max if N>P else 5e-2*MCP_LAMBDA_max
T = 100
MCP_LAMBDA_range = np.flip(np.logspace(np.log10(MCP_LAMBDA_min), np.log10(MCP_LAMBDA_max), num = T))
sdy = np.std(y_train[-1])

update_eta = jax.jit(update_eta_pre)
update_lam = jax.jit(update_lam_pre)

max_iters = 10000
#max_iters = 5
print("Warn low maxitres.")
#max_iters = 100
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
#t, MCP_LAMBDA = 0, MCP_LAMBDA_max
t, MCP_LAMBDA = 1, MCP_LAMBDA_range[1]
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
        #eta, preds = update_eta_pre(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds)
        eta, preds = update_eta(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds)
        #lam, lam_it = update_lam(eta, lam, tau_effective, s, max_iters = lam_maxit)
        lam_it = 0
        #print(eta)

        if lam_it == lam_maxit and verbose:
            print("Reached max iters on lam update.")

        #diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])
        diff = jnp.max(jnp.abs(eta_last-eta))


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

if do_cv:
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

print(np.nanmax(np.abs(etas[:,-1,2]-ncv_betas[3,:])))
print(np.nanmax(np.abs(ncv_preds[0,:] - yy_hat[-1][:,0].T)))

print(ncv_betas[:,-1])

#
#Q = get_Q(etas, lams)
#if do_cv:
#    return Q[tau_opti,-1,:].mean(), yy_hat[-1][tau_opti,:]
#else:
#    return Q.mean(), yy_hat[-1]
