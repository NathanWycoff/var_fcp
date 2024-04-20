import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
N = 100
P = 3

X = np.random.normal(size=[N,P])
sigma2 = np.square(0.1)
y = X[:,0] + np.random.normal(scale=sigma2,size=N)

#nu = np.sqrt(2/np.diag(sigma2 * np.linalg.inv(X.T @ X)))
nu = jnp.ones(P)*1e-5

def nu_cost(lognu, eta, X):
    nu = jnp.exp(lognu)
    x2 = jnp.sum(jnp.square(X), axis=0)
    #a = nu * x2 / (2 * sigma2)
    a = x2/(jnp.square(nu)*sigma2)
    b = -nu * jnp.exp(-nu * jnp.abs(eta))
    c = jnp.log(nu)
    return jnp.sum((2*sigma2)/x2*(a + b + c))


nu_cost_jit = jax.jit(nu_cost)
nu_grad = jax.jit(jax.grad(nu_cost))
nu_h = jax.jit(jax.grad(lambda x,y,z: jnp.sum(nu_grad(x,y,z))))
#nu_h = jax.hessian(nu_cost)

newton_iters = 10

iters = 100

eta = np.zeros(P)

for i in range(iters):
    for p in range(P):
        pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
        resid_other = y - pred_other
        xdn2 = np.sum(np.square(X[:,p]))
        ols = jnp.sum(X[:,p] * resid_other) / xdn2
        b = sigma2*nu[p]/xdn2
        c = nu[p]
        thresh = b*c
        if np.abs(ols) < thresh:
            eta[p] = 0.
        else:
            eta[p] = ols + lambertw(-b*np.square(c)*np.exp(-ols*c))/c
    
    for it in range(newton_iters):
        lognu = jnp.log(nu)
        oldcost = nu_cost(lognu, eta, X)
        step = 1
        newcost = np.inf
        while newcost > oldcost:
            newlognu = lognu - step*nu_grad(lognu, eta, X) / nu_h(lognu, eta, X)
            newcost = nu_cost(newlognu, eta, X)
            step /= 2
        nu = jnp.exp(newlognu)

    print(eta)
    print(nu)
