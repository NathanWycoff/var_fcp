from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

#TODO:
# How many newton iters for nu?

def _nu_cost(lognu, eta, X, tau):
    nu = jnp.exp(lognu)
    x2 = jnp.sum(jnp.square(X), axis=0)#MOVE elsewhere.
    s = sigma2 / x2

    costs = jnp.square(nu)/s - tau*jnp.exp(-jnp.abs(eta)/nu)/2 -jnp.log(nu)

    return jnp.sum(costs)

def _update_eta(eta, nu, X, y, sigma2, tau, P):
    #N,P = X.shape #TOOD: self reference.
    #TODO: precompute squared norm of X variables.

    for p in range(P):
        eo = jnp.delete(eta, p)
        Xo = jnp.delete(X, p, axis=1)
        #print('-----')
        #print('eo:')
        #print(eo)
        #print('Xo:')
        #print(Xo)
        #print('-----')
        pred_other = Xo @ eo
        resid_other = y - pred_other
        xdn2 = jnp.sum(jnp.square(X[:,p]))
        ols = jnp.sum(X[:,p] * resid_other) / xdn2
        s = sigma2 / xdn2
        thresh = (s*tau)/(2*nu[p])

        #print('-----')
        #print("eta:")
        #print(eta)
        #print('ip:')
        #print(jnp.sum(X[:,p] * resid_other))
        #print('mean x:')
        #print(xdn2)
        #print('ols:')
        #print(ols)
        #print('thresh')
        #print(thresh)
        #print('-----')

        true_pred = lambda: 0.
        false_pred = lambda: ols + jnp.sign(ols) * (nu[p] * lambertw(-(s*tau)/(2*jnp.square(nu[p])) * jnp.exp(-jnp.abs(ols)/nu[p])))
        eta_new = jax.lax.cond(jnp.abs(ols) < thresh, true_pred, false_pred)
        eta = eta.at[p].set(eta_new)

    return eta

def _nu_body_fun(val):
    lognu, eta, X, step, newcost, oldcost, tau = val
    h = nu_h(lognu, eta, X, tau)
    #assert np.all(h>=0)
    newlognu = lognu - step*nu_grad(lognu, eta, X, tau) / h
    newcost = _nu_cost(newlognu, eta, X, tau)
    step /= 2
    return newlognu, eta, X, step, newcost, oldcost, tau

def _nu_cond_fun(val):
    eps = 1e-10
    minstep = 1e-8
    lognu, eta, X, step, newcost, oldcost, tau = val
    return jax.lax.cond(step < minstep, lambda: False, lambda: newcost > oldcost - eps*jnp.abs(oldcost))

def _update_nu(eta, nu, X, sigma2, tau, newton_iters):
    for it in range(newton_iters):
        lognu = jnp.log(nu)
        oldcost = _nu_cost(lognu, eta, X, tau)
        step = 1
        newcost = np.inf
        initval = (lognu, eta, X, step, newcost, oldcost, tau)
        val = initval
        val = jax.lax.while_loop(_nu_cond_fun, _nu_body_fun, initval)
        newlognu, eta, X, step, newcost, oldcost, tau = val
        nu = jnp.exp(newlognu)
    
    return nu

nu_grad = jax.grad(_nu_cost)
#nu_h = jax.grad(lambda a,b,c,d: jnp.sum(nu_grad(a,b,c,d)))
nu_h = jax.grad(lambda *args: jnp.sum(nu_grad(*args)))

class SblNet():
    def __init__(self, X, y, sigma2, tau0 = 1, newton_iters = 10, verbose = True, eta_init = None, nu_init = None):
        self.N, self.P = X.shape
        self.tau = tau0*np.sqrt(self.N)

        self.X = X
        self.y = y
        self.sigma2 = sigma2
        self.newton_iters = newton_iters
        self.verbose = verbose 

        self.init_eta(eta_init)
        self.init_nu(nu_init)

        if self.verbose:
            print("Compiling eta update...")
        #self.eta_jit = jax.jit(_update_eta)
        self.eta_jit = _update_eta
        self.eta_jit(self.eta, self.nu, self.X, self.y, self.sigma2, self.tau, self.P)

        if self.verbose:
            print("Compiling nu update...")
        #self.nu_cost_jit = jax.jit(_nu_cost)
        #self.nu_jit = jax.jit(_update_nu, static_argnums = [5])
        self.nu_jit = _update_nu
        self.nu_jit(self.eta, self.nu, self.X, self.sigma2, self.tau, self.newton_iters)

    def init_eta(self, eta_init = None):
        if eta_init is None:
            eta_init = jnp.zeros(self.P)
        self.eta = jnp.array(np.array(eta_init).astype(float))

    def init_nu(self, nu_init = None):
        if nu_init is None:
            nu_init = jnp.sqrt(jnp.diag(sigma2 * jnp.linalg.inv(X.T @ X))/2)
        self.nu = jnp.array(np.array(nu_init).astype(float))

    def _update_Q(self):
        self.Q = tfp.distributions.Laplace(self.eta, self.nu)

    def fit(self, thresh=1e-12, iters=1000):
        for i in range(iters):
            new_eta = self.eta_jit(self.eta, self.nu, self.X, self.y, self.sigma2, self.tau, self.P)
            #print(new_eta)
            diff = jnp.sum(jnp.square(new_eta-self.eta))
            self.eta = new_eta
            print("Not updating nu!")
            #self.nu = self.nu_jit(self.eta, self.nu, self.X, self.sigma2, self.tau, self.newton_iters)

            if diff < thresh:
                if self.verbose:
                    print("Converged after %s iters!"%i)
                break
        
        if i == iters-1:
            print("Did not converge!")

        self._update_Q()

    def ci(self, conf = 0.95):
        alpha = 1-conf
        qq = np.tile([alpha/2,1-alpha/2],[self.P,1]).T
        return self.Q.quantile(qq)

    def summary(self):
        print("Point estimates:")
        print(self.eta)
        print("95\% CI:")
        print(self.ci())
        print("One-sided P Value Analogue:")
        p = self.Q.cdf(0.)
        p = jnp.minimum(p,1-p)
        print(p)

#@partial(jax.jit, static_argnames=['P'])
#@partial(jax.jit, static_argnames=['newton_iters'])
