#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ncvreg_wrapper.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

## Structured selection with hierarchical models a la Roth and Fischer
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr

test_funcs = {}
r = robjects.r

ncvreg = importr('ncvreg')

def pred_ncv(X, y, XX):
    X_arr = robjects.FloatVector(X.T.flatten())
    X_R = robjects.r['matrix'](X_arr, nrow = X.shape[0])

    XX_arr = robjects.FloatVector(XX.T.flatten())
    XX_R = robjects.r['matrix'](XX_arr, nrow = XX.shape[0])

    y_arr = robjects.FloatVector(y)
    y_R = robjects.r['matrix'](y_arr, ncol = 1)

    fit = ncvreg.cv_ncvreg(X_R, y_R)
    beta_hat = r.coef(fit)
    yy_hat = r.predict(fit, XX_R)

    return beta_hat, yy_hat
