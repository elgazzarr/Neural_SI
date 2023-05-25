import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx



class System_A(eqx.Module):
    J: jnp.ndarray # NxN matrix representing recurrent connections between latent states, where N is the number of states
    B: jnp.ndarray # NxC+1 Input projection matrix, which projects inputs u with dimension C concatenated with time as a channel to the dynamcial system
    b: float # bias term
    tau: float # Constant timescale of each neuron 
    N: int # Number of neurons
    C: int # Dimension of input

    def __init__(self, N, C, key):
        super().__init__()
        """ Vector field of system A
            Args:
                N: Number of states
                C: Input dimension
                noise_scale: noise scale
                key: key
        """
        keys = jrandom.split(key,4)
        self.J = jrandom.normal(key=keys[0], shape=(N,N)) * jrandom.bernoulli(key=keys[0], p=0.7, shape=(N,N))
        self.B = jrandom.normal(key=keys[1], shape=(N,C+1)) * jrandom.bernoulli(key=keys[1], p=0.7, shape=(N,C+1))
        self.b = jrandom.uniform(key=keys[2], shape=(N,)) * jrandom.bernoulli(key=keys[2], p=0.7, shape=(N,))
        self.tau = jrandom.uniform(key=keys[2], shape=(N,), minval=0.8, maxval=1.0)
        self.N = N
        self.C = C
    
    def __call__(self, t, x, args):
        u = args.evaluate(t)
        x =  -x + self.J@jnn.tanh(x) + self.B@u + self.b 
        return x/(self.tau + 1e-4)


class System_B(eqx.Module):
    J: jnp.ndarray # NxN matrix representing recurrent connections between latent states, where N is the number of states
    B: jnp.ndarray # NxC+1 Input projection matrix, which projects inputs u with dimension C concatenated with time as a channel to the dynamcial system
    b: float # bias term
    tau: float # Constant timescale of each neuron 
    N: int # Number of neurons
    C: int # Dimension of input


    def __init__(self, N, C, key):
        super().__init__()
        """ Vector field of system B
            Args:
                N: Number of states
                C: Input dimension
                noise_scale: noise scale
                key: key
        """
        keys = jrandom.split(key,4)
        self.J = jrandom.normal(key=keys[0], shape=(N,N)) * jrandom.bernoulli(key=keys[0], p=0.7, shape=(N,N))
        self.B = jrandom.normal(key=keys[1], shape=(N,C+1)) * jrandom.bernoulli(key=keys[1], p=0.7, shape=(N,C+1))
        self.b = jrandom.uniform(key=keys[2], shape=(N,)) * jrandom.bernoulli(key=keys[2], p=0.7, shape=(N,))
        self.tau = jrandom.uniform(key=keys[2], shape=(N,), minval=0.8, maxval=1.0)
        self.N = N
        self.C = C
    
    def __call__(self, t, x, args):
        u = args.evaluate(t)
        x =  -x + jnn.softplus(self.J@x + self.B@u + self.b)
        return x/(self.tau + 1e-4)
    


class Readout(eqx.Module):
    linear_spikes: eqx.nn.Linear
    linear_obs: eqx.nn.Linear
    key: int

    def __init__(self, N, M, O, key):
        super().__init__()
        self.linear_spikes = eqx.nn.Linear(N, M, key=key)
        self.linear_obs = eqx.nn.Linear(M, O, key=key)
        self.key = key
    
    def __call__(self, x):
        rates =  jnn.sigmoid(self.linear_spikes(x))
        spikes = jrandom.poisson(self.key, rates)
        obs = self.linear_obs(rates)
        return rates, spikes, obs