#exec(open('python/ks_term.py').read())
exec(open('python/ks_lib.py').read())

np.random.seed(123)

sigma2 = 1.
alpha = 0.05

N = 100
P = 100 
#P = 500 #Seems like compilation vv slow?
X = np.random.normal(size=[N,P])
y = X[:,0] + np.random.normal(scale=np.sqrt(sigma2), size = N)
beta_true = np.array([1.]+[0. for _ in range(P-1)])

sbl = SblNet(X, y, sigma2 = sigma2)
sbl.fit()

print(sbl.eta)

sbl.summary()

mse = np.sum(np.square(sbl.eta - beta_true))
ci = sbl.ci()

cover = np.logical_or(beta_true > ci[0,:], beta_true < ci[1,:])
np.mean(cover)



# %%
msg = "Hello World"
print(msg)

# %%

def cool(yeboi):
    if yeboi > 0:
        print(yeboi)
    else:
        print('no!!')