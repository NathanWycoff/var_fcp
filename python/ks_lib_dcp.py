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

np.random.seed(123)

N = 100
P = 3

def lam_h(lam, eta, tau):
    return jnp.sum(tau/2*jnp.exp(-lam*jnp.abs(eta)) - jnp.log(lam))
def lam_g(lam, s):
    return jnp.sum(0.5*s/jnp.square(lam))
def lam_cost(lam, eta, tau, s):
    return lam_g(lam,s) - lam_h(lam,eta,tau)

lam_hp = jax.jit(jax.grad(lam_h))

eta = jnp.array([0.8, -4])
tau = 1.2
s = jnp.array([0.1, 1.])


#### Pure python implementation.
##def update_lam(lam, eta, tau, s):
#lam_last = jnp.ones(2)
#it = 0
#diff = np.inf
#while it < max_iters and diff > thresh:
#    it += 1
#    hplam = lam_hp(lam_last, eta, tau)
#    lam = jnp.power(s/(-hplam), 1./3)
#    diff = jnp.max(np.abs(lam_last-lam))
#    lam_last = lam
#    print(diff)
##return lam

## Jax implementation
def body_fun(val):
    lam, eta, tau, s, diff, thresh, it, max_iters = val
    hplam = lam_hp(lam, eta, tau)
    new_lam = jnp.power(s/(-hplam), 1./3)
    diff = jnp.max(jnp.abs(new_lam-lam))
    it1 = it + 1
    return new_lam, eta, tau, s, diff, thresh, it1, max_iters

def cond_fun(val):
    lam, eta, tau, s, diff, thresh, it, max_iters = val
    return (diff > thresh) and (it < max_iters)

def update_lam(lam, eta, tau, s, thresh = 1e-6, max_iters = 100):
    lam = jnp.ones(2)
    it = 0
    thresh = 1e-6

    diff = np.inf
    val = (lam, eta, tau, s, diff, thresh)
    lam, eta, tau, s, diff, thresh = jax.lax.while_loop(cond_fun, body_fun, val)
    return lam

#lam_grid = np.linspace(0,2,num=1000)
#h_grid = np.array([nu_cost(l,eta,tau,s) for l in lam_grid])
#fig = plt.figure()
#plt.plot(lam_grid, h_grid)
#plt.vlines(lam,0,2)
#plt.ylim(0,2)
#plt.savefig("temp.pdf")
#plt.close()




##### OLD:
iters = 10
newton_iters = 10

X = np.random.normal(size=[N,P])
sigma2 = np.square(1)
y = X[:,0] + np.random.normal(scale=sigma2,size=N)
beta_true = np.repeat(0,P)
beta_true[0] = 1.

nu = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X))/2)

eta = jnp.zeros(P)

def update_eta(eta, nu, X, y, sigma2):
    N,P = X.shape #TOOD: self reference.

    for p in range(P):
        pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
        resid_other = y - pred_other
        xdn2 = jnp.sum(jnp.square(X[:,p]))
        ols = jnp.sum(X[:,p] * resid_other) / xdn2
        s = sigma2 / xdn2
        thresh = (s*tau)/(2*nu[p])

        true_pred = lambda: 0.
        false_pred = lambda: ols + jnp.sign(ols) * (nu[p] * lambertw(-(s*tau)/(2*jnp.square(nu[p])) * jnp.exp(-jnp.abs(ols)/nu[p])))
        eta_new = jax.lax.cond(jnp.abs(ols) < thresh, true_pred, false_pred)
        eta = eta.at[p].set(eta_new)

    return eta

def body_fun(val):
    lognu, eta, X, step, newcost, oldcost = val
    h = nu_h(lognu, eta, X)
    #assert np.all(h>=0)
    newlognu = lognu - step*nu_grad(lognu, eta, X) / h
    newcost = nu_cost(newlognu, eta, X)
    step /= 2
    return newlognu, eta, X, step, newcost, oldcost 

def cond_fun(val):
    lognu, eta, X, step, newcost, oldcost = val
    return newcost > oldcost

def update_nu(eta, nu, X, sigma2):
    for it in range(newton_iters):
        lognu = jnp.log(nu)
        oldcost = nu_cost(lognu, eta, X)
        step = 1
        newcost = np.inf
        initval = (lognu, eta, X, step, newcost, oldcost)
        val = initval
        val = jax.lax.while_loop(cond_fun, body_fun, initval)
        newlognu, eta, X, step, newcost, oldcost = val
        nu = jnp.exp(newlognu)
    
    return nu

eta_jit = jax.jit(update_eta)
nu_jit = jax.jit(update_nu)

nu_jit(eta, nu, X, sigma2)

for i in range(iters):
    eta = eta_jit(eta, nu, X, y, sigma2)
    nu = nu_jit(eta, nu, X, sigma2)

    print(eta)
    print(nu)

