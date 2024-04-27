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
from scipy.optimize import minimize_scalar
from python.tfp_plus import tri_quant

import numpy as np
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv

def variational_cost(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
    t1 = N/2*jnp.log(sigma2) # Loglik entropy term
    t2a = jnp.sum(jnp.square(y-X@eta)) # Expected log-lik pred deviation
    t2b = v_f * jnp.sum(jnp.sum(jnp.square(X), axis = 0) / jnp.square(lam)) # Expected log-lik var term.
    t2 = 1/(2*sigma2)*(t2a+t2b) # Expected negative log lik.
    t3 = jnp.sum(tau*jax.vmap(P_FCP)(lam*jnp.abs(eta))) # scaled KS divergence.
    t4 = jnp.sum(jnp.log(lam))
    return t1 + t2 + t3 + t4

################################################################################################
## Us
def pred_sbl(X, y, XX = None, penalty = 'MCP', add_intercept = True, scale = True, verbose = False, do_cv = True, novar = False, plotname = 'traj.pdf', doplot = True):
    #penalty = 'MCP'; add_intercept = True; scale = True; verbose = False; do_cv = False; novar = False; plotname = 'traj.pdf'
    A_MCP = 3.
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
            #interp = jnp.sign(x)*(jnp.abs(x)-s)/(1.-s)
            #smol_s = jnp.minimum(x,jnp.maximum(0.,interp))
            ## Protect against division by 0 in case s=1.
            #big_s = jax.lax.cond(jnp.abs(x)<s, lambda: 0., lambda: x)
            #ret = jax.lax.cond(s<1., lambda: smol_s, lambda: big_s)
            #return ret
            print("Warning: MCP prox ignore equality edge cases.")
            #s *= 2 # Because this is a cdf.

            isgtlt = (jnp.abs(x) > s).astype(int)
            isgtt = (jnp.abs(x)>=1).astype(int)
            #isgtt = (jnp.abs(z)>=1).astype(int)*(s<1).astype(int)
            ind = isgtlt + isgtt
            branches = []
            branches.append(lambda: 0.)
            branches.append(lambda: jnp.sign(x)*(jnp.abs(x)-s)/(1-s))
            branches.append(lambda: x)
            ret3 = jax.lax.switch(ind, branches)

            #ret2 = jax.lax.cond(x<1., lambda: 0., lambda: x)
            ret2 = jax.lax.cond(x<jnp.sqrt(s), lambda: 0., lambda: x)

            ret = jax.lax.cond(s<1., lambda: ret3, lambda: ret2)

            return ret

        P_FCP = lambda x: 0.5*jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.abs(x)-jnp.square(x), lambda: 1.)
        #P_FCP = lambda x: jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.abs(x)-jnp.square(x), lambda: 1.)
        #dP_FCP = lambda x: jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.sign(x)-2*x, lambda: 0.)

        get_Q = lambda eta, lam: tfp.distributions.Triangular(low=eta-1/lam, high=eta+1/lam, peak=eta)
    else:
        raise Exception("Unknown Penalty")

    # Extract other features of penalty.
    #dP_FCP = jax.vmap(jax.grad(P_FCP))
    dP_FCP = jax.vmap(jax.vmap(jax.grad(P_FCP)))
    #print("auto dP_FCP may be unreliable at 0.")
    v_f = get_Q(0,1).variance()

    ## Lambda update functions.
    def body_fun_lam(val):
        eta, lam, tau_effective, s, diff, thresh, it, max_iters = val
        #new_lam = jnp.power(v_f/(s*(eta*tau_effective*dP_FCP(lam*eta)+1/lam)), 1./3)
        new_lam = jnp.power(v_f/((eta*tau_effective*dP_FCP(lam*eta)+s/lam)), 1./3)
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

    #def prox_MCP_manual_biga(z, tau, lam):
    #    #a = 1/(jnp.square(lam)*tau)
    #    #gamma = lam*tau
    #    tau *= 2
    #    isgtlt = (jnp.abs(z) > lam*tau).astype(int)
    #    isgtt = (jnp.abs(z)>=1/lam).astype(int)*(lam*tau<1/lam)
    #    ind = isgtlt + isgtt
    #    branches = []
    #    branches.append(lambda: 0.)
    #    branches.append(lambda: jnp.sign(z)*(jnp.abs(z)-lam*tau)/(1-jnp.square(lam)*tau))
    #    branches.append(lambda: z)
    #    ret = jax.lax.switch(ind, branches)
    #    return ret

    #z = -0.04687909
    #tau = 0.0344094
    #lam = 3.11243891
    #z = 0.10709716
    #tau = 0.02992754
    #lam = 3.33736622

    #z = ols
    #tau = tau_effective[k]
    #lam = lam[k,p]
    #prox_MCP_manual(ols,tau_effective[k],lam[k,p])

    #s = jnp.square(lam)*tau
    #x = lam*z
    #prox_P(x, s) / lam

    #x = ols*lam[k,p]
    #s = jnp.square(lam[k,p])*tau_effective[k]
    #prox_P(x, s)

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
            eta_new = prox_P(ols*lam[k,p], jnp.square(lam[k,p])*tau_effective[k])/lam[k,p]
            #print("Warning: manual prox")
            #assert penalty=='MCP'
            #eta_new_new = prox_MCP_manual(ols, tau_effective[k], lam[k,p])
            #eta_new = eta_new_old
            #eta_new = eta_new_new
            #if abs(eta_new_old-eta_new_new) > 1e-4:
            #    import IPython; IPython.embed()
            eta = eta.at[k,p].set(eta_new)

            preds[k] = pred_other + eta[k,p] * X_train[k][:,p]

        return eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train

    def update_eta_pre(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds):
        val = (eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train)
        eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = jax.lax.fori_loop(0, P, body_fun_eta, val)
        #for p in range(P):
        #    val = body_fun_eta(p,val)
        #eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = val

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
    sdy = np.std(y_train[-1])

    update_eta = jax.jit(update_eta_pre)
    update_lam = jax.jit(update_lam_pre)

    max_iters = 10000
    #max_iters = 5
    #max_iters = 100
    #block_thresh = 1e-6
    block_thresh = 1e-4

    x2 = jnp.array([jnp.sum(jnp.square(Xk), axis=0) for Xk in X_train]) # this is just N when scaled. 

    max_nnz = 40

    eta = jnp.zeros([K,P])
    #MCP_LAMBDA_max = 0.5*np.max(np.abs(X_train[-1].T @ y_train[-1]))/N # Evaluate range on full dataset.
    MCP_LAMBDA_max = np.max(np.abs(X_train[-1].T @ y_train[-1]))/N # Evaluate range on full dataset.

    ## Generate penalty sequence.
    T = 100
    sigma2_hat = jnp.array([np.var(yk) for yk in y_train])
    s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?
    if novar:
        MCP_LAMBDA_min = 1e-3*MCP_LAMBDA_max if N>P else 5e-2*MCP_LAMBDA_max
        #print("Small T")
        #T = 4
        MCP_LAMBDA_range = np.flip(np.logspace(np.log10(MCP_LAMBDA_min), np.log10(MCP_LAMBDA_max), num = T))
        tau_range = A_MCP*np.square(MCP_LAMBDA_range)
    else:
        lam_maxit = 100

        #def rangefind(tau):
        #    lam_eta0 = jnp.sqrt(v_f/s[0,0])
        #    if jnp.square(lam_eta0)*tau < 1.:
        #        ret = np.square(jnp.square(lam[0,0])*tau - MCP_LAMBDA_max)
        #    else:
        #        ret = np.square(jnp.sqrt(tau) - MCP_LAMBDA_max)
        #    #return ret
        #    return ret

        lam_eta0 = jnp.sqrt(v_f/s[0,0])
        def rangefind(tau):
            is_hard = jnp.square(lam_eta0)*tau > 1.
            big_M = 1e4
            ret = np.square(jnp.sqrt(tau) - MCP_LAMBDA_max)
            return ret + big_M*(1-is_hard)

        opt = minimize_scalar(rangefind)
        tau_max = opt.x * (1+1e-1)
        #tau_max = opt.x 
        #tau_max = opt.x
        #tau_max = opt.x*1.5
        #tau_max = 1e10
        tau_min = 1e-3*tau_max if N>P else 5e-2*tau_max
        tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))

        fig = plt.figure()
        tt = np.logspace(-4,2,num=10000)
        ee = [rangefind(t) for t in tt]
        plt.plot(tt, ee)
        plt.xscale('log')
        plt.yscale('log')
        ll,ul = plt.gca().get_ylim()
        plt.vlines(tau_max, ll, ul)
        plt.savefig("temp.pdf")
        plt.close()

        lam = jnp.ones([K,P]) * lam_eta0

        #cost = lambda lam: v_f/(2*s[0]*np.square(lam))+np.log(lam)
        #fig = plt.figure()
        #tt = np.logspace(-4,2)
        #ee = [cost(t) for t in tt]
        #plt.plot(tt, ee)
        #plt.xscale('log')
        #plt.yscale('log')
        #ll,ul = plt.gca().get_ylim()
        #plt.vlines(aopt, ll, ul, color = 'red')
        #plt.vlines(lam[0,0], ll, ul, color = 'blue')
        #plt.savefig("temp.pdf")
        #plt.close()

        #def rangefind(tau):
        #    lam, lam_it = update_lam(jnp.zeros([K,P]), np.ones([K, P]), tau, s, max_iters = lam_maxit)
        #    #lam, _ = update_lam(jnp.zeros(), 1., tau, s[0], max_iters = lam_maxit)
        #    #if lam[0,0] > 1:
        #    #    ret = np.square(1/lam[0,0] - MCP_LAMBDA_max)
        #    #else:
        #    #    ret = np.square(lam[0,0]*tau - MCP_LAMBDA_max)
        #    #return ret
        #    return lam[0,0]
        #fig = plt.figure()
        #tt = np.logspace(-4,2)
        #ee = [rangefind(t) for t in tt]
        #plt.plot(tt, ee)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.savefig("temp.pdf")
        #plt.close()
        #opt = minimize_scalar(rangefind)
        #tau_max = opt.x * (1+1e-1)
        ##tau_max = opt.x
        ##tau_max = opt.x*1.5
        ##tau_max = 1e10
        #tau_min = 1e-3*tau_max if N>P else 5e-2*tau_max
        #tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))
        #lam, lam_it = update_lam(jnp.zeros([K,P]), np.ones([K, P]), tau_max, s, max_iters = lam_maxit)

    ## Init params

    #tau_effective = jnp.array([a*jnp.square(MCP_LAMBDA) for _ in range(K)])
    etas = np.zeros([T, K, P])*np.nan
    lams = np.zeros([T, K, P])*np.nan
    lams_a = np.zeros([T, K, P])*np.nan

    Ns = jnp.array([Xk.shape[0] for Xk in X_train])
    NNs = jnp.array([Xk.shape[0] for Xk in X_test])
    yy_hat = [np.zeros([T,NNs[k]])*np.nan for k in range(K)]
    preds = [X_train[k] @ eta[k,:] for k in range(K)]

    it = 0
    #for t, tau in enumerate(tqdm(tau_range)):
    #t, MCP_LAMBDA = 0, MCP_LAMBDA_max
    t, tau = 0, tau_range[0]
    #t, MCP_LAMBDA = 1, MCP_LAMBDA_range[1]
    #for t, MCP_LAMBDA in enumerate(tqdm(MCP_LAMBDA_range)):
    #    #tau_effective = tau*Ns/Ns[-1]
    #    tau_effective = jnp.array([a*jnp.square(MCP_LAMBDA) for _ in range(K)])
    #    if novar:
    #        #a = 3.
    #        a = 2*3.
    #        lam = np.ones([K, P]) * 1/(a*MCP_LAMBDA)
    for t, tau in enumerate(tqdm(tau_range)):
        tau_effective = tau*jnp.ones(K)

        lam_a = np.ones([K, P]) * 1/jnp.sqrt(A_MCP*tau)
        if novar:
            lam = lam_a


        init_cost = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
        diff = np.inf
        while (it < max_iters) and (diff > block_thresh*sdy):
            it += 1
            eta_last = jnp.copy(eta)
            lam_last = jnp.copy(lam)
            eta, preds = update_eta(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds)
            new_cost = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
            #variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
            #variational_cost(X_train[-1], y_train[-1], eta_last[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
            print("eta:")
            print(init_cost-new_cost)
            if new_cost>init_cost+1e-4:
                import IPython; IPython.embed()
            init_cost = new_cost

            if not novar:
                #import IPython; IPython.embed()
                #sigma2_hat = jnp.array([np.var(yk) for yk in y_train])
                #np.mean(np.square(y_train[k] - preds[k]))
                #sigma2_hat = jnp.array([np.sum(np.square(y_train[k] - preds[k])) / (Ns[k]-np.sum(eta[k,:]!=0)-1) for k in range(K)])
                #sigma2_hat = jnp.array([np.mean(np.square(y_train[k] - preds[k])) for k in range(K)])

                #sigma2_hat = []
                #for k in range(K):
                #    ny = jnp.sum(jnp.square(y_train[k]))
                #    ct = -2 * y_train[k].T @ X_train[k] @ eta[k,:]
                #    qf1 = eta[k,:].T @ X_train[k].T @ X_train[k] @ eta[k,:]
                #    qf2 = v_f * jnp.sum(jnp.sum(jnp.square(X_train[k]), axis = 0) / jnp.square(lam[k,:]))
                #    sigma2_hat.append((ny+ct+qf1+qf2)/N)
                #sigma2_hat = jnp.array(sigma2_hat)
                sigma2_hat = []
                for k in range(K):
                    qf1 = jnp.sum(jnp.square(y_train[k]-preds[k]))
                    qf2 = v_f * jnp.sum(jnp.sum(jnp.square(X_train[k]), axis = 0) / jnp.square(lam[k,:]))
                    sigma2_hat.append((qf1+qf2)/N)
                sigma2_hat = jnp.array(sigma2_hat)

                #if np.any(~np.isfinite(sigma2_hat)):
                #    raise Exception("Bad sigma2!")
                s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?

                new_cost = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
                print("sigma2:")
                print(init_cost-new_cost)
                if new_cost>init_cost+1e-4:
                    import IPython; IPython.embed()
                init_cost = new_cost

                lam, lam_it = update_lam(eta, lam, tau, s, max_iters = lam_maxit)
                print("lam:")
                print(init_cost-new_cost)
                if new_cost>init_cost+1e-4:
                    import IPython; IPython.embed()
                init_cost = new_cost
                if lam_it == lam_maxit and verbose:
                    print("Reached max iters on lam update.")

            diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])
            #diff = jnp.max(jnp.abs(eta_last-eta))

        etas[t,:,:] = eta
        lams[t,:,:] = lam
        lams_a[t,:,:] = lam_a
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
        #tau_opt = tau_range[tau_opti]

    if scale:
        for k in range(K):
            yy_hat[k] = sig_y[k]*yy_hat[k] + mu_y[k]

            etas[:,k,:] = (sig_y[k] / sig_X[k])[np.newaxis,:] * etas[:,k,:]
            lams[:,k,:] = np.square(sig_X[k] / sig_y[k])[np.newaxis,:] * lams[:,k,:]

    NV_plot = np.min([5,P])
    K_plot = np.min([5,P])

    ntnz = np.sum(etas[:,-1,:]!=0, axis = 0)
    top_vars = np.argpartition(ntnz, -K_plot)[-K_plot:]

    cols = [matplotlib.colormaps['tab20'](i) for i in range(K_plot)]

    if doplot:
        Q = get_Q(etas[:,-1,:], lams[:,-1,:])
        if penalty=='MCP':
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
        plt.plot(tau_range, np.delete(etas[:,-1,:], dontplot, axis = 1), color = 'gray')
        plt.xscale('log')
        plt.savefig(plotname)
        plt.close()

    #print(np.nanmax(np.abs(etas[:,-1,2]-ncv_betas[3,:])))
    #print(np.nanmax(np.abs(ncv_preds[0,:] - yy_hat[-1][:,0].T)))
    #
    #print(ncv_betas[:,-1])

    Q = get_Q(etas, lams)
    if do_cv:
        return Q[tau_opti,-1,:].mean(), yy_hat[-1][tau_opti,:]
    else:
        return Q.mean(), yy_hat[-1]

if __name__=='__main__':
    np.random.seed(124)
    N = 400
    P = 40
    X = np.random.normal(size=[N,P])
    #y = np.random.normal(size=[N])
    y = X[:,0] + np.random.normal(size=N)
    XX = np.random.normal(size=[N,P])

    ncv_betas, ncv_preds = pred_ncv_no_cv(X, y, XX)
    sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = False)

    print(np.nanmax(np.abs(sbl_betas[:,-1,2]-ncv_betas[3,:])))
    print(np.nanmax(np.abs(ncv_preds[0,:] - sbl_preds[:,0].T)))
