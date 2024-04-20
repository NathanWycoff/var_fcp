
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
from tqdm import tqdm


def noviss(X, y, sigma2, nu_init = 1e-5, jit = True, max_iters = 100, conv_thresh = 1e-12):
    x2 = np.sum(np.square(X), axis = 0)
    N,P = X.shape

    def nu_cost(lognu, eta, x2, sigma2):
        nu = jnp.exp(lognu)
        a = x2/(jnp.square(nu)*sigma2)
        b = -nu * jnp.exp(-nu * jnp.abs(eta))
        c = jnp.log(nu)
        return jnp.sum((2*sigma2)/x2*(a + b + c))

    def eta_update(eta, nu, X, x2, y, sigma2):
        for p in range(P):
            eta = eta.at[p].set(0)
            #pred_other = jnp.delete(X, p, axis=1) @ jnp.delete(eta, p)
            pred_other = X @ eta
            resid_other = y - pred_other
            ols = jnp.sum(X[:,p] * resid_other) / x2[p]
            b = sigma2*nu[p]/x2[p]
            c = nu[p]
            thresh = b*c
            nz_val = ols + lambertw(-b*jnp.square(c)*jnp.exp(-ols*c))/c
            eta = eta.at[p].set((jnp.abs(ols) > thresh) * nz_val)
        return eta

    def nu_update(eta, nu, x2, sigma2, newton_iters = 10):
        for it in range(newton_iters):
            lognu = jnp.log(nu)
            oldcost = nu_cost(lognu, eta, x2, sigma2)
            step = 1
            newcost = np.inf
            eps = 1e-12
            while newcost > oldcost:
                h = nu_h(lognu, eta, x2, sigma2)
                #newlognu = lognu - step*nu_grad(lognu, eta, x2, sigma2) / (jnp.abs(h)+eps)
                newlognu = lognu - step*nu_grad(lognu, eta, x2, sigma2) / h
                newcost = nu_cost(newlognu, eta, x2, sigma2)
                step /= 2
            print(jnp.min(h))
            nu = jnp.exp(newlognu)
        return nu

    nu_grad = jax.grad(nu_cost)
    nu_h = jax.grad(lambda x,y,z,zz: jnp.sum(nu_grad(x,y,z,zz)))

    if jit:
        nu_cost= jax.jit(nu_cost)
        nu_grad= jax.jit(nu_grad)
        nu_h = jax.jit(nu_h)
        eta_update = jax.jit(eta_update)


    #nu = np.sqrt(2/np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    #nu = jnp.ones(P)*nu_init
    nu = jnp.ones(P)*nu_init
    eta = jnp.zeros(P)

    diff = np.inf
    it = 0
    for it in tqdm(range(max_iters)):
        it += 1

        old_eta = np.array(eta)

        eta = eta_update(eta, nu, X, x2, y, sigma2)
        print(eta)

        #ia = jnp.sqrt(x2/sigma2)
        #ib = 1/(eta+1e-12)
        #nu_init = jnp.minimum(ia,ib)
        ##nu_init = jnp.sqrt(ia*ib)

        nu = jnp.array(nu_init)
        nu = nu_update(eta, nu, x2, sigma2)

        diff = np.sum(np.square(eta-old_eta))
        if diff < conv_thresh:
            break

        print(nu)

    return {'eta':eta,'nu':nu}

N = 100
P = 3

X = np.random.normal(size=[N,P])
sigma2 = np.square(0.1)
np.random.seed(123)
y = X[:,0] + np.random.normal(scale=sigma2,size=N)
noviss(X, y, sigma2, nu_init = 1e-5, jit = False)