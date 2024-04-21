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

np.random.seed(123)

N = 1000
P = 3
#N = 10000
#P = 1000

#penalty = 'laplace'
penalty = 'MCP'

### FCP/Variational Specification
if penalty=='laplace':
    def prox_P(x, s):
        true_pred = lambda: 0.
        false_pred = lambda: x + jnp.sign(x) * lambertw(-s * jnp.exp(-jnp.abs(x)))
        ret = jax.lax.cond(jnp.abs(x) < s, true_pred, false_pred)
        return ret

    P_FCP = lambda x: -jnp.exp(-jnp.abs(x))
    dP_FCP = lambda x: jnp.sign(x)*jnp.exp(-jnp.abs(x))

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

if dP_FCP is None:
    dP_FCP = jax.grad(P_FCP)
v_f = get_Q(0,1).variance()

###

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

X = np.random.normal(size=[N,P])
sigma2_true = np.square(1)
y = X[:,0] + np.random.normal(scale=sigma2_true,size=N)
beta_true = np.repeat(0,P)
beta_true[0] = 1.

def body_fun_eta(p, val):
    eta, lam, tau, s, sigma2, preds = val
    pred_other = preds - eta[p] * X[:,p]
    resid_other = y - pred_other
    xdn2 = sigma2/s[p]
    ols = jnp.sum(X[:,p] * resid_other) / xdn2

    eta_new = prox_P(ols*lam[p], s[p]*jnp.square(lam[p])*tau)/lam[p]
    eta = eta.at[p].set(eta_new)

    preds = pred_other + eta[p] * X[:,p]

    return eta, lam, tau, s, sigma2, preds

def update_eta_pre(eta, lam, X, y, sigma2, tau, s, preds):
    N,P = X.shape 

    val = (eta, lam, tau, s, sigma2, preds)
    eta, lam, tau, s, sigma2, preds  = jax.lax.fori_loop(0, P, body_fun_eta, val)

    return eta, preds

block_iters = 100
block_thresh = 1e-6

update_eta = jax.jit(update_eta_pre)
update_lam = jax.jit(update_lam_pre)

sigma2_hat = np.mean(np.square(y))
x2 = jnp.sum(jnp.square(X), axis=0)
s = sigma2_hat / x2

max_nnz = 40

## Get tau_max
tau_max = np.max(np.abs(X.T @ y))
tau_min = 1e-4
T = 100
tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))
## First order stationarity.

## Init params
eta = jnp.zeros(P)
lam = 1/np.sqrt(s)

etas = np.zeros([T, P])*np.nan
lams = np.zeros([T, P])*np.nan
sigma2s = np.zeros(T)*np.nan

X = jnp.array(X)
y = jnp.array(y)

assert np.all(eta==0)
preds = jnp.zeros(N)

for t, tau in enumerate(tqdm(tau_range)):
    it = 0
    diff = np.inf
    while (it < block_iters) and (diff > block_thresh):
        it += 1
        eta_last = jnp.copy(eta)
        lam_last = jnp.copy(lam)
        eta, preds = update_eta(eta, lam, X, y, sigma2_hat, tau, s, preds)
        lam = update_lam(eta, lam, tau, s)

        diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])

    etas[t,:] = eta
    lams[t,:] = lam

    ## Update variance estimate.
    sigma2_hat = np.mean(np.square(y-preds))
    s = sigma2_hat / x2
    sigma2s[t] = sigma2_hat

    nnz = np.sum(eta!=0)
    if nnz >= max_nnz:
        break

K_plot = np.min([5,P])

ntnz = np.sum(etas!=0, axis = 0)
top_vars = np.argpartition(ntnz, -K_plot)[-K_plot:]

cols = [matplotlib.colormaps['tab20'](i) for i in range(K_plot)]

Q = get_Q(etas, lams)
#lb = Q.quantile(0.025)
#ub = Q.quantile(0.975)
#med = Q.quantile(0.5)
print("Warning: not quantiles!")
med = Q.mean()
lb = Q.mean() - 2*jnp.sqrt(Q.variance())
ub = Q.mean() + 2*jnp.sqrt(Q.variance())

fig = plt.figure()
for vi,v in enumerate(top_vars):
    plt.plot(tau_range, med[:,v], color = cols[vi])
    plt.plot(tau_range, ub[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
    plt.plot(tau_range, lb[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
plt.plot(tau_range, np.delete(etas, top_vars, axis = 1), color = 'gray')
plt.xscale('log')
plt.savefig("traj.pdf")
plt.close()

