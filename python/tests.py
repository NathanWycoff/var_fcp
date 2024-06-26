#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  tests.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.07.2024


# Verify that: 
# 1) P_FCP(0) = 0.
# 2) prox operator agrees.
# 3) Q density agree with derivative of P_FCP?

#def prox_MCP_manual(z, gamma, a):
#    print("Warning: if a=1 we in bad shape!")
#    isgtlt = (jnp.abs(z) > gamma).astype(int)
#    isgtt = (jnp.abs(z)>a*gamma).astype(int)
#    ind = isgtlt + isgtt
#    branches = []
#    branches.append(lambda: 0.)
#    branches.append(lambda: jnp.sign(z)*(jnp.abs(z)-gamma)/(1-1/a))
#    branches.append(lambda: z)
#    ret = jax.lax.switch(ind, branches)
#
#    return ret
