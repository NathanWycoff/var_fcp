import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import equinox as eqx

#TODO:
# How many newton iters for nu?

class SblNet(eqx.Module):
    N: int
    P: int
    tau: float
    X: jnp.array
    y: jnp.array
    eta: jnp.array
    nu: jnp.array
    sigma2: float
    newton_iters: int
    verbose: bool
    #init_eta: 
    eta_jit: callable
    nu_cost_jit: callable
    nu_grad: callable
    nu_h: callable
    nu_jit: callable
    Q: tfp.distributions.Distribution

    def __init__(self, X, y, sigma2, tau0 = 1, newton_iters = 10, verbose = True, eta_init = None, nu_init = None):
        self.N, self.P = X.shape
        self.tau = tau0*np.sqrt(self.N)

        self.X = X
        self.y = y
        self.sigma2 = sigma2
        self.newton_iters = newton_iters
        self.verbose = verbose 

        self.eta = jnp.zeros(self.P)
        self.init_eta(eta_init)
        self.nu = jnp.ones(self.P)
        self.init_nu(nu_init)

        if self.verbose:
            print("Compiling eta update...")
        self.eta_jit = jax.jit(self._update_eta)
        self.eta_jit(self.eta, self.nu, self.X, self.y, self.sigma2)
        self._update_Q()

        if self.verbose:
            print("Compiling nu update...")
        self.nu_cost_jit = jax.jit(self._nu_cost)
        self.nu_grad = jax.jit(jax.grad(self._nu_cost))
        self.nu_h = jax.jit(jax.grad(lambda x,y,z: jnp.sum(self.nu_grad(x,y,z))))
        self.nu_jit = jax.jit(self._update_nu)
        self.nu_jit(self.eta, self.nu, self.X, self.sigma2)

    def init_eta(self, eta_init = None):
        if eta_init is None:
            eta_init = jnp.zeros(self.P)
        self.eta.at[:].set(jnp.array(np.array(eta_init).astype(float)))

    def init_nu(self, nu_init = None):
        if nu_init is None:
            nu_init = jnp.sqrt(jnp.diag(sigma2 * jnp.linalg.inv(X.T @ X))/2)
        self.nu.at[:].set(np.array(np.array(nu_init).astype(float)))

    def _update_Q(self):
        self.Q = tfp.distributions.Laplace(self.eta, self.nu)

    def fit(self, thresh=1e-12, iters=1000):
        for i in range(iters):
            new_eta = self.eta_jit(self.eta, self.nu, self.X, self.y, self.sigma2)
            diff = jnp.sum(jnp.square(new_eta-self.eta))
            #self.eta = new_eta
            self.eta.at[:].set(new_eta)
            self.nu.at[:].set(self.nu_jit(self.eta, self.nu, self.X, self.sigma2))
            #self.nu = self.nu_jit(self.eta, self.nu, self.X, self.sigma2)

            if diff < thresh:
                if self.verbose:
                    print("Converged after %s iters!"%i)
                break
        
        if i == iters-1:
            print("Did not converge!")

        #self._update_Q()

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

    def _nu_cost(self, lognu, eta, X):
        nu = jnp.exp(lognu)
        x2 = jnp.sum(jnp.square(X), axis=0)#MOVE elsewhere.
        s = sigma2 / x2

        costs = jnp.square(nu)/s - self.tau*jnp.exp(-jnp.abs(eta)/nu)/2 -jnp.log(nu)

        return jnp.sum(costs)

    def _update_eta(self, eta, nu, X, y, sigma2):
        #N,P = X.shape #TOOD: self reference.
        #TODO: precompute squared norm of X variables.

        for p in range(self.P):
            pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
            resid_other = y - pred_other
            xdn2 = jnp.sum(jnp.square(X[:,p]))
            ols = jnp.sum(X[:,p] * resid_other) / xdn2
            s = sigma2 / xdn2
            thresh = (s*self.tau)/(2*nu[p])

            true_pred = lambda: 0.
            false_pred = lambda: ols + jnp.sign(ols) * (nu[p] * lambertw(-(s*self.tau)/(2*jnp.square(nu[p])) * jnp.exp(-jnp.abs(ols)/nu[p])))
            eta_new = jax.lax.cond(jnp.abs(ols) < thresh, true_pred, false_pred)
            eta = eta.at[p].set(eta_new)

        return eta

    def _nu_body_fun(self, val):
        lognu, eta, X, step, newcost, oldcost = val
        h = self.nu_h(lognu, eta, X)
        #assert np.all(h>=0)
        newlognu = lognu - step*self.nu_grad(lognu, eta, X) / h
        newcost = self._nu_cost(newlognu, eta, X)
        step /= 2
        return newlognu, eta, X, step, newcost, oldcost 

    def _nu_cond_fun(self, val):
        eps = 1e-10
        minstep = 1e-8
        lognu, eta, X, step, newcost, oldcost = val
        return jax.lax.cond(step < minstep, lambda: False, lambda: newcost > oldcost - eps*jnp.abs(oldcost))

    #self=sbl
    #eta = self.eta
    #nu = self.nu
    def _update_nu(self, eta, nu, X, sigma2):
        for it in range(self.newton_iters):
            lognu = jnp.log(nu)
            oldcost = self._nu_cost(lognu, eta, X)
            step = 1
            newcost = np.inf
            initval = (lognu, eta, X, step, newcost, oldcost)
            val = initval
            val = jax.lax.while_loop(self._nu_cond_fun, self._nu_body_fun, initval)
            newlognu, eta, X, step, newcost, oldcost = val
            nu = jnp.exp(newlognu)
        
        return nu