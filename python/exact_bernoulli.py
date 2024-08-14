#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  exact_bernoulli.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.05.2024

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
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
#exec(open('python/bernoulli.py').read())
from python.bernoulli import pred_sbl

N = 100
NN = 1000
P = 10

np.random.seed(123) # Q: how come we have so much less regularization than them?
#np.random.seed(1234) # We do much better than them.
#np.random.seed(1237) # Q: Our estimate is big: bigger than accurate, bigger than ncv_reg, and even bigger than a straight-up logistic regression!
#np.random.seed(1241) 

sigma = 1.

lik = 'bernoulli'
#lik = 'gaussian'

## Compare on binomial data.
X = np.random.normal(size=[N,P])
XX = np.random.normal(size=[NN,P])

#beta_true = np.array([0.5] + [0 for _ in range(P-1)])
beta_true = np.array([0.4] + [0 for _ in range(P-1)])
if lik=='gaussian':
    y = X@beta_true + np.random.normal(scale=sigma,size=N)
    yy = XX@beta_true + np.random.normal(scale=sigma,size=NN)
elif lik=='bernoulli':
    mu_y = X@beta_true 
    mu_yy = XX@beta_true 
    p_y = jax.nn.sigmoid(mu_y)
    p_yy = jax.nn.sigmoid(mu_yy)
    y = np.random.binomial(1,p=p_y)
    yy = np.random.binomial(1,p=p_yy)
else:
    raise Exception("Bad lik")

## Functiond efs.
def prox_P(x, s):

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

get_Q = lambda eta, lam: tfp.distributions.Triangular(low=eta-1/lam, high=eta+1/lam, peak=eta)
v_f = get_Q(0,1).variance()

key = jax.random.key(0)

if lik=='gaussian':
    def variational_cost(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
        N = X.shape[0]
        t1 = N/2.*jnp.log(sigma2) # Loglik entropy term
        t2a = jnp.sum(jnp.square(y-X@eta)) # Expected log-lik pred deviation
        t2b = v_f * jnp.sum(jnp.sum(jnp.square(X), axis = 0) / jnp.square(lam)) # Expected log-lik var term.
        t2 = 1./(2.*sigma2)*(t2a+t2b) # Expected negative log lik.
        t3 = jnp.sum(tau/sigma2*jax.vmap(P_FCP)(lam*eta)) # scaled KS divergence. 
        t4 = jnp.sum(jnp.log(lam))
        return t1 + t2 + t3 + t4
    def variational_cost_smooth(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
        N = X.shape[0]
        t1 = N/2.*jnp.log(sigma2) # Loglik entropy term
        t2a = jnp.sum(jnp.square(y-X@eta)) # Expected log-lik pred deviation
        t2b = v_f * jnp.sum(jnp.sum(jnp.square(X), axis = 0) / jnp.square(lam)) # Expected log-lik var term.
        t2 = 1./(2.*sigma2)*(t2a+t2b) # Expected negative log lik.
        t4 = jnp.sum(jnp.log(lam))
        return t1 + t2 + t4
elif lik=='bernoulli':
    def variational_cost(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
        Q = get_Q(eta, lam)
        beta_samp = Q.sample(M, seed = key)
        N = X.shape[0]
        t1 = -y.T @ X @ eta
        t2 = jnp.mean(jnp.sum(jax.nn.softplus(X @ beta_samp.T), axis = 0)) # E_q[log(1+e^xb)].
        t3 = jnp.sum(tau*jax.vmap(P_FCP)(lam*eta)) # scaled KS divergence. 
        t4 = jnp.sum(jnp.log(lam))
        return t1 + t2 + t3 + t4
    def variational_cost_smooth(X, y, eta, lam, tau, sigma2, v_f, P_FCP):
        Q = get_Q(eta, lam)
        beta_samp = Q.sample(M, seed = key)
        N = X.shape[0]
        t1 = -y.T @ X @ eta
        t2 = jnp.mean(jnp.sum(jax.nn.softplus(X @ beta_samp.T), axis = 0)) # E_q[log(1+e^xb)].
        t4 = jnp.sum(jnp.log(lam))
        return t1 + t2 + t4
## Functiond efs.

vng = jax.value_and_grad(variational_cost, argnums = [2,3])

M = 30
tau = 1.

np.random.seed(123)
fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = False, novar = False, cost_checks = True, lik = lik, tau_range = [tau], scale = False, sigma2_fixed = np.square(sigma))
eta_me = fst_Q.mean().squeeze()
lam_me = 1/jnp.sqrt(fst_Q.variance()/0.16666).squeeze()

eta = jnp.copy(eta_me)
lam = jnp.copy(lam_me)

#variational_cost(X, y, eta, lam, tau, jnp.square(sigma), v_f, P_FCP)
#grad(X, y, eta, lam, tau, jnp.square(sigma), v_f, P_FCP)

iters = 2000
ss = 1e-4

minlam = 1e-4

eta_traj = np.zeros([iters,P])
lam_traj = np.zeros([iters,P])

costs = np.zeros(iters)
for i in tqdm(range(iters)):
    cost, grad = vng(X, y, eta, lam, tau, jnp.square(sigma), v_f, P_FCP)
    cc = variational_cost(X, y, eta, lam, tau, jnp.square(sigma), v_f, P_FCP)
    #costs[i] = cost
    costs[i] = cc
    eta = eta - ss * grad[0]
    lam = lam - ss * grad[0]
    eta_traj[i,:] = eta
    lam_traj[i,:] = lam
    if np.any(lam)<minlam:
        print("Warning: hitting min.")
        lam = jnp.maximum(minlam, lam)

fig = plt.figure()
plt.subplot(1,3,1)
plt.plot(costs)
plt.subplot(1,3,2)
plt.plot(eta_traj)
plt.subplot(1,3,3)
plt.plot(jnp.log(lam_traj))
plt.savefig('temp.pdf')
plt.close()

variational_cost(X, y, eta, lam, tau, jnp.square(sigma), v_f, P_FCP)
variational_cost(X, y, eta_me, lam_me, tau, jnp.square(sigma), v_f, P_FCP)
