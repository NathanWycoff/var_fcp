#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  dcp_lib_generic.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.21.2024

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

from python.tfp_plus import tri_quant

def pred_sbl(X, y, XX = None, penalty = 'laplace', plotname = 'traj.pdf', add_intercept = True, scale = True):
    N,P = X.shape
    X = jnp.array(X)
    y = jnp.array(y)
    if XX is not None:
        XX = jnp.array(XX)
        NN,PP = XX.shape
        assert PP==P
    else:
        XX = jnp.array(np.nan)
    if scale:
        eps_div = 1e-6
        mu_X = np.mean(X, axis = 0)
        sig_X = np.std(X, axis = 0)+ eps_div
        mu_y = np.mean(y) 
        sig_y = np.std(y) + eps_div
        X = (X - mu_X[np.newaxis,:]) / sig_X[np.newaxis,:]
        XX = (XX - mu_X[np.newaxis,:]) / sig_X[np.newaxis,:]
        y = (y - mu_y) / sig_y

    if add_intercept:
        X = np.concatenate([np.ones([N,1]), X], axis = 1)
        if XX is not None:
            XX = np.concatenate([np.ones([XX.shape[0],1]), XX], axis = 1)
        P += 1

    # do CV splits.
    cv_folds = 5
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
        eta, lam, tau, s, diff, thresh, it, max_iters = val
        new_lam = jnp.power(v_f/(s*(eta*dP_FCP(lam*eta)+1/lam)), 1./3)
        diff = jnp.max(jnp.abs(new_lam-lam))
        return eta, new_lam, tau, s, diff, thresh, it+1, max_iters

    def cond_fun_lam(val):
        eta, lam, tau, s, diff, thresh, it, max_iters = val
        return jnp.logical_and(diff > thresh, it<max_iters)

    def update_lam_pre(eta, lam, tau, s, thresh = 1e-6, max_iters = 100):
        it = 0
        thresh = 1e-6

        diff = np.inf
        val = (eta, lam, tau, s, diff, thresh, 0, max_iters)
        eta, lam, tau, s, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun_lam, body_fun_lam, val)
        return lam

    ## eta update functions
    def body_fun_eta(p, val):
        eta, lam, tau, s, sigma2, preds, X_train, y_train = val
        for k in range(K):
            pred_other = preds[k] - eta[k,p] * X_train[k][:,p]
            resid_other = y_train[k] - pred_other
            xdn2 = sigma2[k]/s[k,p]
            ols = jnp.sum(X_train[k][:,p] * resid_other) / xdn2

            eta_new = prox_P(ols*lam[k,p], s[k,p]*jnp.square(lam[k,p])*tau[k])/lam[k,p]
            eta = eta.at[k,p].set(eta_new)

            preds[k] = pred_other + eta[k,p] * X_train[k][:,p]

        return eta, lam, tau, s, sigma2, preds, X_train, y_train

    def update_eta_pre(eta, lam, X_train, y_train, sigma2, tau, s, preds):
        val = (eta, lam, tau, s, sigma2, preds, X_train, y_train)
        eta, lam, tau, s, sigma2, preds, X_train, y_train = jax.lax.fori_loop(0, P, body_fun_eta, val)

        return eta, preds

    block_iters = 100
    block_thresh = 1e-6

    update_eta = jax.jit(update_eta_pre)
    update_lam = jax.jit(update_lam_pre)

    #sigma2_hat = np.mean(np.square(y))
    sigma2_hat = jnp.array([np.mean(np.square(yk)) for yk in y_train])
    #x2 = jnp.sum(jnp.square(X), axis=0)
    x2 = jnp.array([jnp.sum(jnp.square(Xk), axis=0) for Xk in X_train])
    s = sigma2_hat[:,jnp.newaxis] / x2

    max_nnz = 40

    ## Get tau_max
    tau_max = np.max(np.abs(X.T @ y)) # Evaluate range on full dataset.
    tau_min = 1e-4
    T = 100
    tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))
    ## First order stationarity.

    ## Init params
    eta = jnp.zeros([K,P])
    lam = jnp.array(1/np.sqrt(s))

    etas = np.zeros([T, K, P])*np.nan
    lams = np.zeros([T, K, P])*np.nan
    sigma2s = np.zeros([T,K])*np.nan

    Ns = jnp.array([Xk.shape[0] for Xk in X_train])
    NNs = jnp.array([Xk.shape[0] for Xk in X_test])
    yy_hat = [np.zeros([T,NNs[k]])*np.nan for k in range(K)]
    preds = [X_train[k] @ eta[k,:] for k in range(K)]

    for t, tau in enumerate(tqdm(tau_range)):
        tau_effective = tau*Ns/Ns[-1]
        it = 0
        diff = np.inf
        while (it < block_iters) and (diff > block_thresh):
            it += 1
            eta_last = jnp.copy(eta)
            lam_last = jnp.copy(lam)
            eta, preds = update_eta(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds)
            lam = update_lam(eta, lam, tau_effective, s)

            diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])

        etas[t,:,:] = eta
        lams[t,:,:] = lam
        for k in range(K):
            yy_hat[k][t,:] = X_test[k] @ eta[k,:]

        ## Update variance estimate.
        sigma2_hat = jnp.array([np.mean(np.square(y_train[k]-preds[k])) for k in range(K)])
        s = sigma2_hat[:,jnp.newaxis] / x2
        sigma2s[t,:] = sigma2_hat

        nnz = np.sum(eta[-1,::]!=0)
        if nnz >= max_nnz:
            break

    ### Get rid of any we didn't actually do.
    #etas = etas[:t,:]
    #lams = lams[:t,:]
    #yy_hat = yy_hat[:t,:]
    #tau_range = tau_range[:t]

    errs = jnp.zeros(T)
    for k in range(K-1):
        errs += np.mean(np.square(yy_hat[k] - y_test[k]), axis = 1) / (K-1)
    tau_opti = np.nanargmin(errs)
    tau_opt = tau_range[tau_opti]

    #X = np.array(X)
    if scale:
        for k in range(K):
            #X[:,1:] = sig_X[np.newaxis,:] * X[:,1:] + mu_X[np.newaxis,:]
            #y = sig_y*y + mu_y 
            yy_hat[k] = sig_y*yy_hat[k] + mu_y 

            int_modif = etas[:,k,1:] @ (mu_X/sig_X)
            #etas = etas.at[:,0].set(sig_y*etas[:,0] + mu_y - sig_y * int_modif)
            #etas = etas.at[:,1:].set(sig_y / sig_X * etas[:,1:])
            etas[:,k,0] = sig_y*etas[:,k,0] + mu_y - sig_y * int_modif
            etas[:,k,1:] = sig_y / sig_X * etas[:,k,1:]
            lams[:,k,0] = lams[:,k,0] / np.square(sig_y)
            lams[:,k,1:] = np.square(sig_X / sig_y) * lams[:,k,1:]

    NV_plot = np.min([5,P])
    k_plot = -1

    ntnz = np.sum(etas[:,k_plot,:]!=0, axis = 0)
    if add_intercept:
        ntnz[0] = 0
    top_vars = np.argpartition(ntnz, -NV_plot)[-NV_plot:]

    cols = [matplotlib.colormaps['tab20'](i) for i in range(NV_plot)]

    Q = get_Q(etas[:,k_plot,:], lams[:,k_plot,:])
    if penalty=='MCP':
        lb = tri_quant(Q, 0.025)
        ub = tri_quant(Q, 0.975)
        med = tri_quant(Q, 0.5)
    else:
        lb = Q.quantile(0.025)
        ub = Q.quantile(0.975)
        med = Q.quantile(0.5)

    fig = plt.figure()
    plt.subplot(2,1,1)
    for vi,v in enumerate(top_vars):
        plt.plot(tau_range, med[:,v], color = cols[vi])
        plt.plot(tau_range, ub[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
        plt.plot(tau_range, lb[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
    dontplot = top_vars
    if add_intercept:
        dontplot = np.concatenate([[0], dontplot])
    plt.plot(tau_range, np.delete(etas[:,k_plot,:], dontplot, axis = 1), color = 'gray')
    plt.xscale('log')
    ll,ul = plt.gca().get_ylim()
    plt.vlines(tau_opt,ll,ul)

    plt.subplot(2,1,2)
    plt.plot(tau_range, errs)
    plt.xscale('log')
    plt.title("Cross Validation Error")
    ll,ul = plt.gca().get_ylim()
    plt.vlines(tau_opt,ll,ul)

    plt.tight_layout()
    plt.savefig(plotname)
    plt.close()

    return Q[tau_opti,:], yy_hat[-1][tau_opti,:]

if __name__=='__main__':
    np.random.seed(123)

    N = 1000
    P = 95
    #N = 10000
    #P = 1000

    X = np.random.normal(size=[N,P])
    XX = np.random.normal(size=[N,P])
    sigma2_true = np.square(1)
    y = X[:,0] + np.random.normal(scale=sigma2_true,size=N) + 50
    yy = XX[:,0] + np.random.normal(scale=sigma2_true,size=N) + 50
    # TODO: should be used in y generation.
    beta_true = np.repeat(0,P)
    beta_true[0] = 1.

    sbl = SblNet(X, y, XX=XX, penalty = 'laplace')
    sbl.Q.mean()[40,:]

    beta_hat, yy_hat = pred_sbl(X, y, XX)

    beta_ols = np.linalg.lstsq(X, y)[0]
    yy_ols = XX @ beta_ols

    print(np.sum(np.square(beta_hat - beta_true)))
    print(np.sum(np.square(beta_ols - beta_true)))

    print(np.sum(np.square(yy_hat - yy)))
    print(np.sum(np.square(yy_ols - yy)))

