#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ks_lib_dcp.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.20.2024

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
import scipy.stats as ss

np.random.seed(123)

N = 1000
P = 3
#N = 10000
#P = 1000

def lam_h(lam, eta, tau):
    return jnp.sum(tau/2*jnp.exp(-lam*jnp.abs(eta)) - jnp.log(lam))
def lam_g(lam, s):
    return jnp.sum(1./(2.*s*jnp.square(lam)))
def lam_cost(lam, eta, tau, s):
    return lam_g(lam,s) - lam_h(lam,eta,tau)

#lam_hp = jax.jit(jax.grad(lam_h))
lam_hp = jax.grad(lam_h)

def body_fun_lam(val):
    eta, lam, tau, s, diff, thresh, it, max_iters = val
    hplam = lam_hp(lam, eta, tau)
    new_lam = jnp.power(1/(-s*hplam), 1./3)
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

X = np.random.normal(size=[N,P])
sigma2 = np.square(1)
y = X[:,0] + np.random.normal(scale=sigma2,size=N)
beta_true = np.repeat(0,P)
beta_true[0] = 1.

def body_fun_eta(p, val):
    eta, lam, tau, s, preds = val
    pred_other = preds - eta[p] * X[:,p]
    #pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
    resid_other = y - pred_other
    xdn2 = jnp.sum(jnp.square(X[:,p]))
    ols = jnp.sum(X[:,p] * resid_other) / xdn2
    #s = sigma2 / xdn2
    thresh = (lam[p]*s[p]*tau)/2

    true_pred = lambda: 0.
    false_pred = lambda: ols + jnp.sign(ols) * (1/lam[p] * lambertw(-(s[p]*tau*jnp.square(lam[p]))/(2) * jnp.exp(-jnp.abs(ols)*lam[p])))
    eta_new = jax.lax.cond(jnp.abs(ols) < thresh, true_pred, false_pred)
    eta = eta.at[p].set(eta_new)

    preds = pred_other + eta[p] * X[:,p]

    return eta, lam, tau, s, preds

def update_eta(eta, lam, X, y, sigma2, tau):
    N,P = X.shape #TOOD: self reference.

    x2 = jnp.sum(jnp.square(X), axis=0)
    s = sigma2 / x2

    preds = X @ eta
    val = (eta, lam, tau, s, preds)
    eta, lam, tau, s, preds  = jax.lax.fori_loop(0, P, body_fun_eta, val)

    return eta

block_iters = 100
block_thresh = 1e-6

eta_jit = jax.jit(update_eta)
lam_jit = jax.jit(update_lam)

x2 = jnp.sum(jnp.square(X), axis=0)
s = sigma2 / x2

max_nnz = 40

tau_max = 1e8
tau_min = 1e-4
T = 100
tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))

## Init params
eta = jnp.zeros(P)
#lam = 1/np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X))/2)
lam = 1/np.sqrt(sigma2 / (2*x2))

etas = np.zeros([T, P])*np.nan
lams = np.zeros([T, P])*np.nan

X = jnp.array(X)
y = jnp.array(y)

for t, tau in enumerate(tqdm(tau_range)):
    it = 0
    diff = np.inf
    while (it < block_iters) and (diff > block_thresh):
        it += 1
        eta_last = jnp.copy(eta)
        lam_last = jnp.copy(lam)
        eta = eta_jit(eta, lam, X, y, sigma2, tau)
        lam = lam_jit(eta, lam, tau, s)

        diff = max([jnp.max(jnp.abs(eta_last-eta)), jnp.max(jnp.abs(lam_last-lam))])

    etas[t,:] = eta
    lams[t,:] = lam

    nnz = np.sum(eta!=0)
    if nnz >= max_nnz:
        break

K_plot = np.min([5,P])

ntnz = np.sum(etas!=0, axis = 0)
top_vars = np.argpartition(ntnz, -K_plot)[-K_plot:]

cols = [matplotlib.colormaps['tab20'](i) for i in range(K_plot)]

fig = plt.figure()
for vi,v in enumerate(top_vars):
    plt.plot(tau_range, etas[:,v], color = cols[vi])
    plt.plot(tau_range, etas[:,v] + 2.3*1/lams[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
    plt.plot(tau_range, etas[:,v] - 2.3*1/lams[:,v], color = cols[vi], linestyle='--', alpha = 0.5)
plt.xscale('log')
plt.savefig("traj.pdf")
plt.close()

