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

#reps = 10

tau = 10

def nu_cost(lognu, eta, X):
    nu = jnp.exp(lognu)
    x2 = jnp.sum(jnp.square(X), axis=0)#MOVE elsewhere.
    s = sigma2 / x2

    costs = jnp.square(nu)/s - tau*jnp.exp(-jnp.abs(eta)/nu)/2 -jnp.log(nu)

    return jnp.sum(costs)

iters = 10
newton_iters = 10

nu_cost_jit = jax.jit(nu_cost)
nu_grad = jax.jit(jax.grad(nu_cost))
nu_h = jax.jit(jax.grad(lambda x,y,z: jnp.sum(nu_grad(x,y,z))))
#nu_h = jax.hessian(nu_cost)

#err_us = np.zeros(reps)
#err_ols = np.zeros(reps)
#for rep in range(reps):

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

#def update_eta(eta, nu, X, y, sigma2):
#    for p in range(P):
#        pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
#        resid_other = y - pred_other
#        xdn2 = np.sum(np.square(X[:,p]))
#        ols = jnp.sum(X[:,p] * resid_other) / xdn2
#        s = sigma2 / xdn2
#        thresh = (s*tau)/(2*nu[p])
#        if jnp.abs(ols) < thresh:
#            print('a')
#            eta = eta.at[p].set(0.) 
#        else:
#            print('b')
#            lambertw_arg = -(s*tau)/(2*jnp.square(nu[p])) * jnp.exp(-jnp.abs(ols)/nu[p])
#            delta = nu[p] * lambertw(lambertw_arg)
#            new_eta = ols + jnp.sign(ols) * delta
#            print(new_eta)
#            eta = eta.at[p].set(new_eta)
#    return eta


#for p in range(P):
#    pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
#    resid_other = y - pred_other
#    xdn2 = np.sum(np.square(X[:,p]))
#    ols = jnp.sum(X[:,p] * resid_other) / xdn2
#    s = sigma2 / xdn2
#    thresh = (s*tau)/(2*nu[p])
#    if jnp.abs(ols) < thresh:
#        eta[p] = 0
#    else:
#        lambertw_arg = -(s*tau)/(2*jnp.square(nu[p])) * jnp.exp(-jnp.abs(ols)/nu[p])
#        delta = nu[p] * lambertw(lambertw_arg)
#        eta[p] = ols + jnp.sign(ols) * delta

#def while_loop_inner(lognu, eta, X, step):
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

#def update_nu(eta, nu, X, sigma2):
#    for it in range(newton_iters):
#        lognu = jnp.log(nu)
#        oldcost = nu_cost(lognu, eta, X)
#        step = 1
#        newcost = np.inf
#        while newcost > oldcost:
#            h = nu_h(lognu, eta, X)
#            assert np.all(h>=0)
#            newlognu = lognu - step*nu_grad(lognu, eta, X) / h
#            newcost = nu_cost(newlognu, eta, X)
#            step /= 2
#        nu = jnp.exp(newlognu)
#    return nu

#update_eta(eta, nu, X, y, sigma2)
#
##eta_jit(eta, nu, X, y, sigma2)
#
#update_nu(eta, nu, X, sigma2)
#

for i in range(iters):
    eta = eta_jit(eta, nu, X, y, sigma2)
    nu = nu_jit(eta, nu, X, sigma2)

    print(eta)
    print(nu)

#    err_us[rep] = np.sum(np.square(eta-beta_true))
#    beta_ols = np.linalg.lstsq(X,y)[0]
#    err_ols[rep] = np.sum(np.square(beta_ols-beta_true))
#
#fig = plt.figure()
##plt.boxplot([err_ols, err_us])
#plt.boxplot(err_ols-err_us)
#plt.savefig("slr_box.pdf")
#plt.close()
np.linalg.lstsq(X[:,0:1],y)[0]
