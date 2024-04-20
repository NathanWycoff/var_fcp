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

np.random.seed(123)

N = 100
P = 3

def lam_h(lam, eta, tau):
    return jnp.sum(tau/2*jnp.exp(-lam*jnp.abs(eta)) - jnp.log(lam))
def lam_g(lam, s):
    return jnp.sum(0.5*s/jnp.square(lam))
def lam_cost(lam, eta, tau, s):
    return lam_g(lam,s) - lam_h(lam,eta,tau)

#lam_hp = jax.jit(jax.grad(lam_h))
lam_hp = jax.grad(lam_h)

## Jax implementation
def body_fun(val):
    lam, eta, tau, s, diff, thresh, it, max_iters = val
    hplam = lam_hp(lam, eta, tau)
    new_lam = jnp.power(s/(-hplam), 1./3)
    diff = jnp.max(jnp.abs(new_lam-lam))
    return new_lam, eta, tau, s, diff, thresh, it+1, max_iters

def cond_fun(val):
    lam, eta, tau, s, diff, thresh, it, max_iters = val
    return jnp.logical_and(diff > thresh, it<max_iters)

def update_lam(eta, lam, tau, s, thresh = 1e-6, max_iters = 100):
    it = 0
    thresh = 1e-6

    diff = np.inf
    val = (lam, eta, tau, s, diff, thresh, 0, max_iters)
    lam, eta, tau, s, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun, body_fun, val)
    return lam

X = np.random.normal(size=[N,P])
sigma2 = np.square(1)
y = X[:,0] + np.random.normal(scale=sigma2,size=N)
beta_true = np.repeat(0,P)
beta_true[0] = 1.


def update_eta(eta, lam, X, y, sigma2, tau):
    N,P = X.shape #TOOD: self reference.

    for p in range(P):
        pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
        resid_other = y - pred_other
        xdn2 = jnp.sum(jnp.square(X[:,p]))
        ols = jnp.sum(X[:,p] * resid_other) / xdn2
        s = sigma2 / xdn2
        thresh = (lam[p]*s*tau)/(2)

        true_pred = lambda: 0.
        false_pred = lambda: ols + jnp.sign(ols) * (1/lam[p] * lambertw(-(s*tau*jnp.square(lam[p]))/(2) * jnp.exp(-jnp.abs(ols)*lam[p])))
        eta_new = jax.lax.cond(jnp.abs(ols) < thresh, true_pred, false_pred)
        eta = eta.at[p].set(eta_new)

    return eta


block_iters = 100
block_thresh = 1e-6

eta_jit = jax.jit(update_eta)
lam_jit = jax.jit(update_lam)

x2 = jnp.sum(jnp.square(X), axis=0)
s = sigma2 / x2

tau_max = 1e8
tau_min = 1e-4
T = 100
tau_range = np.flip(np.logspace(np.log10(tau_min), np.log10(tau_max), num = T))

## Init params
eta = jnp.zeros(P)
lam = 1/np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X))/2)

etas = np.zeros([T, P])
lams = np.zeros([T, P])

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

fig = plt.figure()
plt.plot(tau_range, etas)
plt.xscale('log')
plt.savefig("traj.pdf")
plt.close()
