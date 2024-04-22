#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  tfp_plus.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2024

import jax
import jax.numpy as jnp
import numpy as np

def tri_quant(Q, alpha):
    a = Q.low
    b = Q.high
    c = Q.peak

    h = 1
    m1 = h / (c-a)
    b1 = -a*m1
    m2 = -h / (b-c)
    b2 = -b*m2

    Z1 = m1/2*(np.square(c)-np.square(a)) + b1*(c-a)
    Z2 = m2/2*(np.square(b)-np.square(c)) + b2*(b-c)
    Z = Z1 + Z2

    isbot = alpha <= Z1
    cq_bot = np.square(a)+2*b1/m1*a+2*Z*alpha/m1
    bot = 0.5 * (-2*b1/m1 + np.sqrt(np.square(2*b1/m1)+4*cq_bot))
    cq_top = (alpha*Z - Z1 + np.square(c)*m2/2+b2*c)
    top = (-b2+np.sqrt(np.square(b2)+2*m2*cq_top))/m2
    return isbot * bot + (1-isbot)*top

