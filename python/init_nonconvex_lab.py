from tqdm import tqdm

exec(open('python/ks_lib.py').read())

np.random.seed(123)

sigma2 = 1.
alpha = 0.05

N = 5
P = 2
X = np.random.normal(size=[N,P])
y = X[:,0] + np.random.normal(scale=np.sqrt(sigma2), size = N)
beta_true = np.array([1.]+[0. for _ in range(P-1)])

#sbl = SblNet(X, y, sigma2 = sigma2, verbose = False, eta_init = [6,9])
#sbl.fit()
#sbl.eta

ng = 2
mmax = 3
grid = np.linspace(-mmax, mmax, num = ng)

A = np.zeros([ng,ng])

#reuse = False
reuse = True

if reuse:
    sbl = SblNet(X, y, sigma2 = sigma2, verbose = True)

init = [-3,-3]
print('1--')
print(init)
if reuse:
    sbl.init_eta(init)
    sbl.init_nu()
else:
    sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
sbl.fit(iters=1)
print('--------------')
print(sbl.eta)
print('--------------')

init = [-3,3]
print('2--')
print(init)
if reuse:
    sbl.init_eta(init)
    sbl.init_nu()
else:
    sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
sbl.fit(iters=1)
print('--------------')
print(sbl.eta)
print('--------------')

#if reuse:
#    sbl = SblNet(X, y, sigma2 = sigma2, verbose = True)
#
#for xi, x in enumerate(tqdm(grid)):
#    for yi, y in enumerate(grid):
#        init = [x,y]
#        print('--')
#        print(init)
#        if reuse:
#            sbl.init_eta(init)
#            sbl.init_nu()
#        else:
#            sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
#        sbl.fit(iters=1)
#        print(sbl.eta)
#        val = np.sqrt(np.sum(np.square(sbl.eta)))
#        A[xi,yi] = val
#        #print(init)
#        #print(val)
#
#fig = plt.figure()
#im = plt.imshow(A, origin = 'lower', extent = [-mmax, mmax, -mmax, mmax])
#fig.colorbar(im)
#plt.savefig("init.pdf")
#plt.close()

######## ############### ############### ############### ################### ########
#sbl = SblNet(X, y, sigma2 = sigma2, eta_init = [-3,-3], verbose = False)
#sbl.fit()
#sbl.eta
#
#sbl = SblNet(X, y, sigma2 = sigma2, eta_init = [0,0], verbose = False)
#sbl.fit()
#sbl.eta
#
### With resuse
#for xi, x in enumerate(tqdm(grid)):
#    for yi, y in enumerate(grid):
#        init = [x,y]
#        sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
#        sbl.fit()
#        val = np.sqrt(np.sum(np.square(sbl.eta)))
#        print(init)
#        print(val)
#
#
### With no resuse
#for xi, x in enumerate(tqdm(grid)):
#    for yi, y in enumerate(grid):
#        init = [x,y]
#        sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
#        sbl.fit()
#        val = np.sqrt(np.sum(np.square(sbl.eta)))
#        print(init)
#        print(val)
#
#init = [grid[1], grid[1]]
#sbl = SblNet(X, y, sigma2 = sigma2, verbose = True, eta_init = init)
#sbl.fit()
#sbl.eta
#sbl = SblNet(X, y, sigma2 = sigma2, eta_init = [-3,-3], verbose = False)
#sbl.fit()
#sbl.eta
#
#sbl = SblNet(X, y, sigma2 = sigma2, eta_init = [0,0], verbose = False)
#sbl.fit()
#sbl.eta
#
### With resuse
#for xi, x in enumerate(tqdm(grid)):
#    for yi, y in enumerate(grid):
#        init = [x,y]
#        sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
#        sbl.fit()
#        val = np.sqrt(np.sum(np.square(sbl.eta)))
#        print(init)
#        print(val)
#
#
### With no resuse
#for xi, x in enumerate(tqdm(grid)):
#    for yi, y in enumerate(grid):
#        init = [x,y]
#        sbl = SblNet(X, y, sigma2 = sigma2, eta_init = init, verbose = True)
#        sbl.fit()
#        val = np.sqrt(np.sum(np.square(sbl.eta)))
#        print(init)
#        print(val)
#
#init = [grid[1], grid[1]]
#sbl = SblNet(X, y, sigma2 = sigma2, verbose = True, eta_init = init)
#sbl.fit()
#sbl.eta