#!/usr/bin/env python3
# -*- coding: utf-7 -*-

import jax
jax.config.update("jax_enable_x64", True)
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
from scipy.optimize import minimize_scalar

import numpy as np
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv


def variational_cost(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
    N = X.shape[0]
    t1 = N/2.*jnp.log(sigma2) # Loglik entropy term
    t2a = jnp.sum(jnp.square(y-X@eta)) # Expected log-lik pred deviation
    t2b = v_f * jnp.sum(jnp.sum(jnp.square(X), axis = 0) / jnp.square(lam)) # Expected log-lik var term.
    t2 = 1./(2.*sigma2)*(t2a+t2b) # Expected negative log lik.
    t3 = jnp.sum(tau/sigma2*jax.vmap(P_FCP)(lam*eta)) # scaled KS divergence. 
    t4 = jnp.sum(jnp.log(lam))
    return t1 + t2 + t3 + t4

def lam_costs(lam, eta, X, sigma2, v_f, P_FCP, tau):
    t1 = v_f * jnp.sum(jnp.square(X), axis = 0) / jnp.square(lam)/(2.*sigma2) # Expected log-lik var term.
    t2 = tau/sigma2*jax.vmap(P_FCP)(lam*eta) # scaled KS divergence.
    t3 = jnp.log(lam)
    return t1 + t2 + t3

################################################################################################
## Us
def pred_sbl(X, y, XX = None, penalty = 'MCP', add_intercept = True, scale = True, verbose = True, do_cv = True, novar = False, plotname = 'traj.pdf', doplot = True, cost_checks = False, max_nnz = np.inf):
    assert not (np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isnan(XX)))
    #penalty = 'MCP'; add_intercept = True; scale = True; verbose = False; do_cv = True; novar = False; plotname = 'traj.pdf'; cost_checks = False; max_nnz = np.inf
    if not do_cv:
        print("New version not tested without CV.")
    if novar:
        print("New version not tested without var.")
    if cost_checks:
        print("New version not tested with cost checks.")
    A_MCP = 3.
    lam_maxit = 100
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
            #print("Warning: MCP prox ignore equality edge cases.")
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
            ret2 = jax.lax.cond(jnp.abs(x)<jnp.sqrt(s), lambda: 0., lambda: x)

            ret = jax.lax.cond(s<1., lambda: ret3, lambda: ret2)

            return ret

        P_FCP = lambda x: 0.5*jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.abs(x)-jnp.square(x), lambda: 1.)
        #dP_FCP = jax.vmap(jax.vmap(lambda x: 0.5*jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.sign(x)-2*x, lambda: 0.)))
        #P_FCP = lambda x: jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.abs(x)-jnp.square(x), lambda: 1.)
        #dP_FCP = lambda x: jax.lax.cond(jnp.abs(x)<1, lambda: 2*jnp.sign(x)-2*x, lambda: 0.)

        get_Q = lambda eta, lam: tfp.distributions.Triangular(low=eta-1/lam, high=eta+1/lam, peak=eta)
    else:
        raise Exception("Unknown Penalty")

    # Extract other features of penalty.
    dP_FCP1 = jax.grad(P_FCP)
    dP_FCP = jax.vmap(jax.vmap(dP_FCP1))
    v_f = get_Q(0,1).variance().astype(np.float64)

    prox_Pv = jax.vmap(prox_P)
    P_FCPv = jax.vmap(jax.vmap(P_FCP))

    #def update_sigma2_pre(sigma2_hat, y_train, preds, Ns, eta, lam, v_f, x2, tau):
    #    for k in range(K):
    #        t1 = jnp.sum(jnp.square(y_train[k] - preds[k]))
    #        t2 = v_f * jnp.sum(x2[k,:]/jnp.square(lam[k,:]))
    #        t3 = jnp.sum(2*tau*jax.vmap(P_FCP)(lam[k,:]*eta[k,:]))
    #        sigma2_hat = sigma2_hat.at[k].set((t1+t2+t3)/Ns[k])
    #    return sigma2_hat

    def update_sigma2_pre(sigma2_hat, y_train, preds, Ns, eta, lam, v_f, x2, tau):
        nnz = jnp.sum(eta!=0, axis = 1)
        t1 = jnp.nansum(jnp.square(y_train - preds), axis = 1)
        t2 = v_f * jnp.nansum(x2/jnp.square(lam), axis = 1)
        t3 = jnp.nansum(2*tau*P_FCPv(lam*eta), axis = 1)
        sigma2_hat = (t1+t2+t3)/(Ns+P-nnz)
        return sigma2_hat

    ## Lambda update functions.
    def body_fun_lam(val):
        eta, lam, tau_effective, s, sigma2_wide, diff, thresh, it, max_iters = val
        new_lam = jnp.power(1/s*v_f/(eta*tau_effective[:,np.newaxis]/sigma2_wide*dP_FCP(lam*eta)+1/lam), 1./3)
        diff = jnp.max(jnp.abs(new_lam-lam))
        return eta, new_lam, tau_effective, s, sigma2_wide, diff, thresh, it+1, max_iters

    def cond_fun_lam(val):
        eta, lam, tau_effective, s, sigma2_wide, diff, thresh, it, max_iters = val
        return jnp.logical_and(diff > thresh, it<max_iters)

    def update_lam_pre(eta, lam, tau_effective, s, sigma2_wide, thresh = 1e-6, max_iters = 100):
        it = 0
        diff = np.inf

        val = (eta, lam, tau_effective, s, sigma2_wide, diff, thresh, 0, max_iters)
        eta, lam, tau_effective, s, sigma2_wide, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun_lam, body_fun_lam, val)
        return lam, it

    ## eta update functions
    def body_fun_eta(p, val):
        eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train, x2 = val

        pred_other = preds - eta[:,p,np.newaxis] * X_train[:,:,p]
        resid_other = y_train - pred_other
        ols = jnp.nanmean(X_train[:,:,p]*resid_other, axis = 1)

        eta_new = prox_Pv(ols*lam[:,p], jnp.square(lam[:,p])*tau_effective/x2[:,p])/lam[:,p]
        eta = eta.at[:,p].set(eta_new)

        preds = pred_other + eta[:,p,np.newaxis] * X_train[:,:,p]

        return eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train, x2

    def update_eta_pre(eta, lam, X_train, y_train, x2, sigma2_hat, tau_effective, s, preds):
        val = (eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train, x2)
        eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train, x2 = jax.lax.fori_loop(0, P, body_fun_eta, val)
        return eta, preds
    ########################################

    # do CV splits.
    if do_cv:
        cv_folds = 10
        K = cv_folds+1
        inds = np.arange(N)
        np.random.shuffle(inds)
        test_inds = np.array_split(inds, cv_folds)
        train_inds = [np.setdiff1d(np.arange(N),ti) for ti in test_inds]
        Ns = jnp.array([len(t) for t in train_inds]+[N]) # Size of each train fold, and full dataset is last. 
        NNs = jnp.array([len(t) for t in test_inds]+[NN]) # Size of each test fold

        max_N = np.max(Ns)
        max_NN = np.max(NNs)
        X_train = jnp.zeros([K,max_N,P])*np.nan
        y_train = jnp.zeros([K,max_N])*np.nan
        X_test = jnp.zeros([K,max_NN,P])*np.nan
        y_test = jnp.zeros([K,max_NN])*np.nan

        for cv_i in range(cv_folds):
            testi = test_inds[cv_i]
            traini = np.setdiff1d(np.arange(N),test_inds[cv_i])

            X_train = X_train.at[cv_i,:Ns[cv_i],:].set(X[traini,:])
            y_train = y_train.at[cv_i,:Ns[cv_i]].set(y[traini])
            X_test = X_test.at[cv_i,:NNs[cv_i],:].set(X[testi,:])
            y_test = y_test.at[cv_i,:NNs[cv_i]].set(y[testi])

        X_train = X_train.at[-1,:Ns[-1],:].set(X)
        y_train = y_train.at[-1,:Ns[-1]].set(y)
        X_test = X_test.at[-1,:NNs[-1],:].set(XX)
        y_test = y_test.at[-1,:NNs[-1]].set(np.nan*np.zeros(NN))

    else:
        K = 1
        X_train = X[np.newaxis,:,:]
        y_train = y[np.newaxis,:]
        X_test = XX[np.newaxis,:,:]
        y_test = jnp.zeros(XX.shape[0])*jnp.nan

        Ns = jnp.array([N])
        NNs = jnp.array([NN])

    if scale:
        eps_div = 1e-6
        mu_X = jnp.nanmean(X_train, axis = 1)
        sig_X = jnp.nanstd(X_train, axis = 1) + eps_div
        mu_y = np.nanmean(y_train, axis = 1)

        X_train = (X_train - mu_X[:,np.newaxis,:]) / sig_X[:,np.newaxis,:]
        X_test = (X_test - mu_X[:,np.newaxis,:]) / sig_X[:,np.newaxis,:]

        y_train = (y_train - mu_y[:,np.newaxis])
        y_test = (y_test - mu_y[:,np.newaxis])


    ## Get tau_max
    sdy = np.std(y_train[-1])

    update_eta = jax.jit(update_eta_pre)
    update_lam = jax.jit(update_lam_pre)
    update_sigma2 = jax.jit(update_sigma2_pre)

    max_iters = 10000
    block_thresh = 1e-4

    x2 = jnp.nansum(jnp.square(X_train), axis = 1)

    eta = jnp.zeros([K,P])
    MCP_LAMBDA_max = np.max(np.abs(X_train[-1,:,:].T @ y_train[-1,:]))/N # Evaluate range on full dataset.

    ## Generate penalty sequence.
    T = 100

    #yy_hat = [np.zeros([T,NNs[k]])*np.nan for k in range(K)]
    yy_hat = np.zeros([K,T,max_NN])*np.nan
    #preds = [X_train[k] @ eta[k,:] for k in range(K)]
    preds = (X_train @ eta[:,:,np.newaxis]).squeeze()

    if novar:
        MCP_LAMBDA_min = 1e-3*MCP_LAMBDA_max if N>P else 5e-2*MCP_LAMBDA_max
        MCP_LAMBDA_range = np.flip(np.logspace(np.log10(MCP_LAMBDA_min), np.log10(MCP_LAMBDA_max), num = T))
        tau_range = A_MCP*np.square(MCP_LAMBDA_range)
    else:
        ## Closed form optim
        #ynorm2 = np.array([np.sum(np.square(yt)) for yt in y_train])
        ynorm2 = jnp.nansum(jnp.square(y_train), axis = 1)
        lam = jnp.sqrt(Ns[:,np.newaxis]*v_f*x2/ynorm2[:,np.newaxis])*jnp.ones([K,P])
        sigma2_hat = ynorm2/Ns
        sigma2_wide = sigma2_hat[:,np.newaxis] * jnp.ones([K,P]) #TODO: Sort out exactly what we want wrt sigma and x2.
        s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?

        xmax = np.max(x2[-1,:])
        lam_eta0 = np.min(lam)

        sbig = MCP_LAMBDA_max > 1/lam_eta0
        if sbig:
            tau_max = xmax * jnp.square(MCP_LAMBDA_max)
        else:
            tau_max = xmax * jnp.abs(MCP_LAMBDA_max) / lam_eta0
        tau_max *= (1+1e-1)

        tau_min = 1e-3*tau_max if N>P else 5e-2*tau_max
        tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))

    ## Trace storage
    etas = np.zeros([T, K, P])*np.nan
    lams = np.zeros([T, K, P])*np.nan
    lams_a = np.zeros([T, K, P])*np.nan

    it = 0
    #t, tau = 0, tau_range[0]
    for t, tau in enumerate(tqdm(tau_range, disable = not verbose)):
        tau_effective = tau*jnp.ones(K) # TODO: Remvoe

        lam_a = np.ones([K, P]) * 1/jnp.sqrt(A_MCP*tau)
        if novar:
            lam = lam_a
            sigma2_hat = jnp.array([np.var(yk) for yk in y_train])
            s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?
            sigma2_wide = jnp.array([sigma2_hat[k]*jnp.ones(P) for k in range(K)])

        diff = np.inf
        while (it < max_iters) and (diff > block_thresh*sdy):
            it += 1
            eta_last = jnp.copy(eta)
            lam_last = jnp.copy(lam)
            preds_last = [jnp.copy(p) for p in preds]

            if cost_checks:
                cost_before = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
            eta, preds = update_eta(eta, lam, X_train, y_train, x2, sigma2_hat, tau_effective, s, preds)
            if cost_checks:
                cost_after = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
            if cost_checks and cost_after > cost_before + 1e-8:
                print("It's eta!")
                print(cost_after - cost_before)
                t_broke = t
                import IPython; IPython.embed()

            if not novar:
                if cost_checks:
                    cost_before = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)
                #lc_before =  lam_costs(lam[-1,:], eta[-1,:], X_train[-1], sigma2_hat[-1], v_f, P_FCP)
                lam, lam_it = update_lam(eta, lam, tau_effective, s, sigma2_wide, max_iters = lam_maxit)
                #lc_after =  lam_costs(lam[-1,:], eta[-1,:], X_train[-1], sigma2_hat[-1], v_f, P_FCP)
                if cost_checks:
                    cost_after = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP)

                if cost_checks and cost_after > cost_before + 1e-8:
                    print("It's lam!")
                    print(cost_after - cost_before)
                    print(it)
                    import IPython; IPython.embed()

                if lam_it == lam_maxit and verbose:
                    print("Reached max iters on lam update.")
                if cost_checks:
                    nnz = jnp.sum(eta!=0, axis = 1)
                    cost_before = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP) + (P-nnz)/2*jnp.log(sigma2_hat)
                sigma2_hat = update_sigma2(sigma2_hat, y_train, preds, Ns, eta, lam, v_f, x2, tau)
                #sigma2_wide = jnp.array([sigma2_hat[k]*jnp.ones(P) for k in range(K)])
                sigma2_wide = sigma2_hat[:,np.newaxis] * jnp.ones([K,P]) #TODO: Sort out exactly what we want wrt sigma and x2.
                s = sigma2_hat[:,jnp.newaxis] / x2 # Such that this is just 1/N?
                if cost_checks:
                    cost_after = variational_cost(X_train[-1], y_train[-1], eta[-1,:], lam[-1,:], tau, sigma2_hat[-1], v_f, P_FCP) + (P-nnz)/2*jnp.log(sigma2_hat)
                if cost_checks and cost_after > cost_before + 1e-8:
                    print("It's sigma2!")
                    print(cost_after - cost_before)
                    import IPython; IPython.embed()

            diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])
            #diff = jnp.max(jnp.abs(eta_last-eta))

        etas[t,:,:] = eta
        lams[t,:,:] = lam
        lams_a[t,:,:] = lam_a
        yy_hat[:,t,:] = (X_test @ eta[:,:,np.newaxis]).squeeze()

        nnz = np.sum(eta[-1,:]!=0)
        if nnz >= max_nnz:
            break

    if verbose:
        if it==max_iters:
            print("Reached Max Iters on outer block.")
        #print("Stopped main loop on tau %d after %d iters with %d nnz"%(t,it,np.sum(eta!=0)))

    if do_cv:
        errs = (np.apply_over_axes(np.nansum, np.square(yy_hat-y_test[:,np.newaxis,:]), [0,2]) / (K-1)).squeeze()
        tau_opti = np.nanargmin(errs) # TODO: do we really want nanargmin here?

    if scale:
        yy_hat + mu_y[:,np.newaxis,np.newaxis]
        etas = 1/sig_X[np.newaxis,:,:] * etas 
        lams = np.square(sig_X)[np.newaxis,:,:] * lams

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

    Q = get_Q(etas, lams)
    if do_cv:
        #return Q[tau_opti,-1,:].mean(), yy_hat[-1][tau_opti,:]
        return Q[tau_opti,-1,:], yy_hat[-1,tau_opti,:]
    else:
        #return Q.mean(), yy_hat[-1]
        return Q, yy_hat[-1,:]

#if __name__=='__main__':
#    np.random.seed(124)
#    N = 40
#    P = 40
#    X = np.random.normal(size=[N,P])
#    #y = X[:,0] + np.random.normal(size=N) + 50
#    y = -1.08 * X[:,0] + np.random.normal(size=N) + 50
#    #y = -0.03*X[:,0] + np.random.normal(size=N) + 50
#    XX = np.random.normal(size=[N,P])
#
#    ncv_betas, ncv_preds = pred_ncv_no_cv(X, y, XX)
#    #sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = True)
#    sbl_betas, sbl_preds = pred_sbl(X, y, XX, do_cv = False, novar = False, cost_checks = True)
#
#    #print(np.nanmax(np.abs(sbl_betas[:,-1,2]-ncv_betas[3,:])))
#    print(np.nanmax(np.abs(sbl_betas[:,-1,0]-ncv_betas[1,:])))
#    print(np.nanmax(np.abs(ncv_preds[0,:] - sbl_preds[:,0].T)))
