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

class SblNet(object):
    def __init__(self, X, y, XX = None, penalty = 'laplace', add_intercept = True, scale = True):
        N,P = X.shape
        NN = XX.shape[0]
        if scale:
            eps_div = 1e-6
            self.mu_X = np.mean(X, axis = 0)
            self.sig_X = np.std(X, axis = 0)+ eps_div
            self.mu_y = np.mean(y) 
            self.sig_y = np.std(y) + eps_div
            X = (X - self.mu_X[np.newaxis,:]) / self.sig_X[np.newaxis,:]
            XX = (XX - self.mu_X[np.newaxis,:]) / self.sig_X[np.newaxis,:]
            y = (y - self.mu_y) / self.sig_y

        if add_intercept:
            X = np.concatenate([np.ones([N,1]), X], axis = 1)
            if XX is not None:
                XX = np.concatenate([np.ones([NN,1]), XX], axis = 1)
            P += 1

        self.X = jnp.array(X)
        self.y = jnp.array(y)
        self.XX = jnp.array(XX)
        self.P = P
        self.N = N
        self.NN = NN
        self.scale = scale
        self.add_intercept = add_intercept
        self.penalty = penalty

        NN,PP = XX.shape
        assert PP==P

        ### FCP/Variational Specification
        if penalty=='laplace':
            def prox_P(x, s):
                true_pred = lambda: 0.
                false_pred = lambda: x + jnp.sign(x) * lambertw(-s * jnp.exp(-jnp.abs(x)))
                ret = jax.lax.cond(jnp.abs(x) < s, true_pred, false_pred)
                return ret

            P_FCP = lambda x: -jnp.exp(-jnp.abs(x))
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

        self.prox_P = prox_P
        self.get_Q = get_Q
        # Extract other features of penalty.
        self.dP_FCP = jax.vmap(jax.grad(P_FCP))
        self.v_f = get_Q(0,1).variance()

        ### Init update functions.
        self._create_updates()
        self.update_eta = jax.jit(self.update_eta)
        self.update_lam = jax.jit(self.update_lam)
        
    def _create_updates(self):
        ## Lambda update functions.
        def body_fun_lam(val):
            eta, lam, tau, s, diff, thresh, it, max_iters = val
            new_lam = jnp.power(self.v_f/(s*(eta*self.dP_FCP(lam*eta)+1/lam)), 1./3)
            diff = jnp.max(jnp.abs(new_lam-lam))
            return eta, new_lam, tau, s, diff, thresh, it+1, max_iters

        def cond_fun_lam(val):
            eta, lam, tau, s, diff, thresh, it, max_iters = val
            return jnp.logical_and(diff > thresh, it<max_iters)

        def update_lam(eta, lam, tau, s, thresh = 1e-6, max_iters = 100):
            it = 0
            thresh = 1e-6

            diff = np.inf
            val = (eta, lam, tau, s, diff, thresh, 0, max_iters)
            eta, lam, tau, s, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun_lam, body_fun_lam, val)
            return lam

        ## eta update functions
        def body_fun_eta(p, val):
            eta, lam, tau, s, sigma2, preds, X, y = val
            pred_other = preds - eta[p] * X[:,p]
            resid_other = y - pred_other
            xdn2 = sigma2/s[p]
            ols = jnp.sum(X[:,p] * resid_other) / xdn2

            eta_new = self.prox_P(ols*lam[p], s[p]*jnp.square(lam[p])*tau)/lam[p]
            eta = eta.at[p].set(eta_new)

            preds = pred_other + eta[p] * X[:,p]

            return eta, lam, tau, s, sigma2, preds, X, y

        def update_eta(eta, lam, X, y, sigma2, tau, s, preds):
            val = (eta, lam, tau, s, sigma2, preds, X, y)
            eta, lam, tau, s, sigma2, preds, X, y = jax.lax.fori_loop(0, self.P, body_fun_eta, val)

            return eta, preds

        self.update_eta = update_eta
        self.update_lam = update_lam

    def _cv_select_tauind(self):
        stop("First, you cant use tau indices when the tau_range is different")
        stop("Second, shouldn't we be N-normalizing these guys?")
        cv_folds = 5
        inds = np.arange(self.N)
        np.random.shuffle(inds)
        test_inds = np.array_split(inds, cv_folds)

        errs = list()
        for cv_i in range(cv_folds):
            testi = test_inds[cv_i]
            traini = np.setdiff1d(np.arange(self.N),test_inds[cv_i])

            X_train = self.X[traini,:]
            y_train = self.y[traini]
            X_test = self.X[testi,:]
            y_test = self.y[testi]

            yy_hat = self.fit_predict(X_train, y_train, XX = X_test)
            err = np.mean(np.square(yy_hat - y_test[np.newaxis,:]), axis = 1)
            errs.append(err)

        errs = np.stack(errs).T
        mse_vs_tau = np.mean(errs, axis = 1)
        tau_opti = np.nanargmin(mse_vs_tau)

        return tau_opti

    def cv_fit_predict(self):
        tau_opti = self._cv_select_tauind()
        yy_hat = self.fit_predict()
        return self.Q.mean()[tau_opti,:], yy_hat[tau_opti,:]

    def fit_predict(self, X=None, y=None, XX=None, plotname = 'traj.pdf', plot = False):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if XX is None:
            XX = self.XX

        N,P = X.shape
        NN = XX.shape[0]

        block_iters = 100
        block_thresh = 1e-6

        sigma2_hat = np.mean(np.square(y))
        x2 = jnp.sum(jnp.square(X), axis=0)
        s = sigma2_hat / x2

        #max_nnz = 40
        max_nnz = 200

        ## Get tau_max
        tau_max = np.max(np.abs(X.T @ y))
        tau_min = 1e-4
        T = 100
        tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))
        ## First order stationarity.

        ## Init params
        eta = jnp.zeros(self.P)
        lam = 1/np.sqrt(s)

        etas = np.zeros([T, self.P])*np.nan
        lams = np.zeros([T, self.P])*np.nan
        sigma2s = np.zeros(T)*np.nan
        if XX is not None:
            yy_hat = np.zeros([T,NN])*np.nan

        assert np.all(eta==0)
        preds = jnp.zeros(N)

        for t, tau in enumerate(tqdm(tau_range, leave = False)):
            it = 0
            diff = np.inf
            while (it < block_iters) and (diff > block_thresh):
                it += 1
                eta_last = jnp.copy(eta)
                lam_last = jnp.copy(lam)
                eta, preds = self.update_eta(eta, lam, X, y, sigma2_hat, tau, s, preds)
                lam = self.update_lam(eta, lam, tau, s)

                diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])

            etas[t,:] = eta
            lams[t,:] = lam
            if XX is not None:
                yy_hat[t,:] = XX @ eta

            ## Update variance estimate.
            sigma2_hat = np.mean(np.square(y-preds))
            s = sigma2_hat / x2
            sigma2s[t] = sigma2_hat

            nnz = np.sum(eta!=0)
            if nnz >= max_nnz:
                break

        ### Get rid of any we didn't actually do.
        #etas = etas[:t,:]
        #lams = lams[:t,:]
        #yy_hat = yy_hat[:t,:]
        #tau_range = tau_range[:t]

        #X = np.array(X)
        if self.scale:
            #X[:,1:] = self.sig_X[np.newaxis,:] * X[:,1:] + self.mu_X[np.newaxis,:]
            #y = self.sig_y*y + self.mu_y 
            yy_hat = self.sig_y*yy_hat + self.mu_y 

            int_modif = etas[:,1:] @ (self.mu_X/self.sig_X)
            etas[:,0] = self.sig_y*etas[:,0] + self.mu_y - self.sig_y * int_modif
            etas[:,1:] = self.sig_y / self.sig_X * etas[:,1:]
            lams[:,0] = lams[:,0] / np.square(self.sig_y)
            lams[:,1:] = np.square(self.sig_X / self.sig_y) * lams[:,1:]

        K_plot = np.min([5,self.P])

        ntnz = np.sum(etas!=0, axis = 0)
        if self.add_intercept:
            ntnz[0] = 0
        top_vars = np.argpartition(ntnz, -K_plot)[-K_plot:]

        cols = [matplotlib.colormaps['tab20'](i) for i in range(K_plot)]

        Q = self.get_Q(etas, lams)
        if plot:
            if self.penalty=='MCP':
                lb = tri_quant(Q, 0.025)
                ub = tri_quant(Q, 0.975)
                med = tri_quant(Q, 0.5)
            else:
                lb = Q.quantile(0.025)
                ub = Q.quantile(0.975)
                med = Q.quantile(0.5)

            fig = plt.figure()
            for vi,v in enumerate(top_vars):
                plt.plot(tau_range, med[:,v], color = cols[vi])
                plt.plot(tau_range, ub[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
                plt.plot(tau_range, lb[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
            dontplot = top_vars
            if self.add_intercept:
                dontplot = np.concatenate([[0], dontplot])
            plt.plot(tau_range, np.delete(etas, dontplot, axis = 1), color = 'gray')
            plt.xscale('log')
            plt.savefig(plotname)
            plt.close()

        self.Q = Q
        self.tau_range = tau_range

        return(yy_hat)

def pred_sbl(X, y, XX):
    ## CV
    N,P = X.shape

    cv_folds = 5
    inds = np.arange(N)
    np.random.shuffle(inds)
    test_inds = np.array_split(inds, cv_folds)

    errs = list()
    for cv_i in range(cv_folds):
        testi = test_inds[cv_i]
        traini = np.setdiff1d(np.arange(N),test_inds[cv_i])

        X_train = X[traini,:]
        y_train = y[traini]
        X_test = X[testi,:]
        y_test = y[testi]

        sbl = SblNet(X_train, y_train, XX = X_test, penalty = 'laplace')
        err = np.mean(np.square(sbl.yy_hat - y_test[np.newaxis,:]), axis = 1)
        errs.append(err)

    errs = np.stack(errs).T
    mse_vs_tau = np.mean(errs, axis = 1)
    tau_opti = np.nanargmin(mse_vs_tau)

    ## Rerun on full data
    sbl = SblNet(X, y, XX=XX, penalty = 'laplace')
    return sbl.Q.mean()[tau_opti,:], sbl.yy_hat[tau_opti,:]

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
    #sbl.fit(X, y, XX)
    beta_hat, yy_hat = sbl.cv_fit_predict()

    beta_ols = np.linalg.lstsq(X, y)[0]
    yy_ols = XX @ beta_ols

    print(np.sum(np.square(beta_hat - beta_true)))
    print(np.sum(np.square(beta_ols - beta_true)))

    print(np.sum(np.square(yy_hat - yy)))
    print(np.sum(np.square(yy_ols - yy)))

