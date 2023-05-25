import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx
import diffrax as dfx
from typing import Callable


class Controller(dfx.AbstractPath):

    c: int
    phases: Callable
    frequencies: Callable
    t_stim: float # start time of the stimulus
    key: jrandom.PRNGKey

    def __init__(self, t_stim, ph, freqs, c):
        self.c = c
        self.phases = ph
        self.frequencies = freqs
        self.t_stim = t_stim
        self.key = jrandom.PRNGKey(0)
        
        
    def evaluate(self, t, t1 = None, left = True):
        del left
        controls = jnp.array([jnp.sin(self.phases[i]+ self.frequencies[i]*jnn.relu(t-self.t_stim))*jnn.sigmoid(t-(self.t_stim+1))
                               for i in range(self.c)])
        controls = controls 

        return controls