#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  mcp_scad_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.21.2024

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax.math import lambertw
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


