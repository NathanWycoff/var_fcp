

### FCP/Variational Specification
if penalty=='laplace':
    def prox_P(x, s):
        true_pred = lambda: 0.
        false_pred = lambda: x + jnp.sign(x) * lambertw(-s * jnp.exp(-jnp.abs(x)))
        ret = jax.lax.cond(jnp.abs(x) < s, true_pred, false_pred)
        return ret

    P_FCP = lambda x: -jnp.exp(-jnp.abs(x))
    #dP_FCP = lambda x: jnp.sign(x)*jnp.exp(-jnp.abs(x))

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

# Extract other features of penalty.
#dP_FCP = jax.vmap(jax.grad(P_FCP))
dP_FCP = jax.vmap(jax.vmap(jax.grad(P_FCP)))
v_f = get_Q(0,1).variance()

## Lambda update functions.
def body_fun_lam(val):
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = val
    new_lam = jnp.power(v_f/(s*(eta*dP_FCP(lam*eta)+1/lam)), 1./3)
    diff = jnp.max(jnp.abs(new_lam-lam))
    return eta, new_lam, tau_effective, s, diff, thresh, it+1, max_iters

def cond_fun_lam(val):
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = val
    return jnp.logical_and(diff > thresh, it<max_iters)

def update_lam_pre(eta, lam, tau_effective, s, thresh = 1e-6, max_iters = 100):
    it = 0
    thresh = 1e-6

    diff = np.inf
    val = (eta, lam, tau_effective, s, diff, thresh, 0, max_iters)
    eta, lam, tau_effective, s, diff, thresh, it, max_iters = jax.lax.while_loop(cond_fun_lam, body_fun_lam, val)
    return lam, it

## eta update functions
def body_fun_eta(p, val):
    eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = val
    for k in range(K):
        pred_other = preds[k] - eta[k,p] * X_train[k][:,p]
        resid_other = y_train[k] - pred_other
        xdn2 = sigma2_hat[k]/s[k,p]
        #ols = jnp.sum(X_train[k][:,p] * resid_other) / xdn2
        ols = jnp.mean(X_train[k][:,p] * resid_other)

        eta_new = prox_P(ols*lam[k,p], s[k,p]*jnp.square(lam[k,p])*tau_effective[k])/lam[k,p]
        eta = eta.at[k,p].set(eta_new)

        preds[k] = pred_other + eta[k,p] * X_train[k][:,p]

    return eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train

def update_eta_pre(eta, lam, X_train, y_train, sigma2_hat, tau_effective, s, preds):
    val = (eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train)
    eta, lam, tau_effective, s, sigma2_hat, preds, X_train, y_train = jax.lax.fori_loop(0, P, body_fun_eta, val)

    return eta, preds
