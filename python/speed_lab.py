#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from python.ncvreg_wrapper import pred_ncv, pred_ncv_no_cv
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

np.random.seed(124)
N = 41 # TODO: 41 is way harder randomly?
P = 40
X = np.random.normal(size=[N,P])
#y = X[:,0] + np.random.normal(size=N) + 50
y = -1.08 * X[:,0] + np.random.normal(size=N) + 50
#y = -0.03*X[:,0] + np.random.normal(size=N) + 50
XX = np.random.normal(size=[N,P])

exec(open('python/valencia.py').read())
np.random.seed(123)
val_Q, val_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = False)
val_Q.mean()

exec(open('python/fast_boi.py').read())
np.random.seed(123)
fst_Q, fst_preds = pred_sbl(X, y, XX, do_cv = True, novar = False, cost_checks = False)
fst_Q.mean()

