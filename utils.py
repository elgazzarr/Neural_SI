import jax
import equinox as eqx
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
from typing import Callable, List
import jax.nn as jnn
from jax import jit
import numpy as np 
from jax import Array
from diffrax.custom_types import PyTree, Scalar, Union
import seaborn as sns
import pandas as pd

class Vectorfield_encoder(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)



class Encoder(eqx.Module):
    initial: eqx.nn.MLP
    vf: Vectorfield_encoder

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey = jrandom.split(key, 2)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.vf = Vectorfield_encoder(data_size, hidden_size, width_size, depth, key=fkey)

    @eqx.filter_jit
    def __call__(self, ts, coeffs, get_context):
        control = dfx.LinearInterpolation(ts, coeffs)
        term = dfx.ControlTerm(self.vf, control).to_ode()
        solver = dfx.Tsit5()
        dt0 = (ts[1] - ts[0])/2
        y0 = self.initial(control.evaluate(ts[0]))
        saveat = dfx.SaveAt(ts=ts) if get_context else dfx.SaveAt(t1=True)
        solution = dfx.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
        )
        return solution.ys


class Vectorfield_mlp(eqx.Module):

    state_net: eqx.nn.MLP
    in_net: eqx.nn.MLP
    tau: float

    def __init__(self, latent_size, input_size, width_size, depth,  key):
        if input_size is None:
            self.in_net = None
        else:
            self.in_net = eqx.nn.Linear(input_size, latent_size, key=key)
        self.state_net = eqx.nn.MLP(
            in_size=latent_size,
            width_size=width_size,
            out_size=latent_size,
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=jax.nn.tanh,
            key=key,
        )
        self.tau = jrandom.uniform(key=key, shape=(latent_size,), minval=0.8, maxval=1.2)

    def __call__(self, t, y, u):
        if u is None:
            return self.net(y)
        else:
            return (-y + self.state_net(y) + self.in_net(u.evaluate(t)))/ (self.tau + 1e-6)
    
    
class Vectorfield_wc(eqx.Module):

    J: jnp.ndarray
    B: jnp.ndarray 
    b: float # bias term
    tau: float # time constant

    def __init__(self, latent_size, input_size, *, key):
        super().__init__()
        """ Vector field in latent space
            Args:
                N: Number of states
                C: Input dimension
                key: key
        """
        keys = jrandom.split(key,4)
        self.J = jrandom.normal(key=keys[1], shape=(latent_size,))
        input_size = 0 if input_size is None else input_size
        self.B = jrandom.normal(key=keys[1], shape=(latent_size+1, input_size))
        self.b = jrandom.normal(key=keys[2], shape=(latent_size,))
        self.tau = jrandom.uniform(key=keys[3], shape=(latent_size,))
    
    def __call__(self, t, x, u):
        if u is not None:
            u = u.evaluate(t)
        else:
            u = jnp.zeros((self.B.shape[1],))
        x =  (-x+self.J@jnn.tanh(x) + self.b + self.B@u)
        return x/self.tau




class Vectorfield_mlp_posterior(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, latent_size, input_size, hidden_size, wdith_size, depth, key):

        in_size = input_size if input_size is not None else 0

        self.net = eqx.nn.MLP(
            in_size=latent_size + hidden_size  + in_size,
            width_size=wdith_size,
            out_size=latent_size,
            depth=depth,
            activation=jax.nn.softplus,
            #final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        # Note that the state, input and hidden state are concatenated into y in
        #  sde_kl_utils
        #print(y.shape)
        #jax.debug.print('y shape {}', y.shape)
        return self.net(y)

class Vectorfield_mlp_prior(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, latent_size, input_size, wdith_size, depth, key):

        in_size = input_size if input_size is not None else 0

        self.net = eqx.nn.MLP(
            in_size=latent_size + in_size,
            width_size=wdith_size,
            out_size=latent_size,
            depth=depth,
            activation=jax.nn.softplus,
            #final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.net(jnp.concatenate([y, args.evaluate(t)], axis=-1))


class Diffusion(eqx.Module):

    nets: List[eqx.nn.MLP]

    def __init__(self, latent_size, *, key):

        keys = jrandom.split(key, latent_size)
        self.nets = [
            eqx.nn.MLP(
                in_size=1,
                width_size=16,
                out_size=1,
                depth=1,
                activation=jax.nn.softplus,
                final_activation=jax.nn.sigmoid,
                key=i_key,
            )
            for i_key in keys
        ]

    def __call__(self, t, y, args):
        y = jnp.split(y, indices_or_sections=len(self.nets))
        out = [net_i(y_i) for net_i, y_i in zip(self.nets, y)]
        return jnp.concatenate(out, axis=0)


class Diffusion_mlp(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, latent_size, *, key):

        self.net = eqx.nn.MLP(
            in_size=latent_size,
            width_size=32,
            out_size=latent_size,
            depth=1,
            activation=jax.nn.tanh,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.net(y)



class Readout(eqx.Module):
    linear_spikes: eqx.nn.Linear
    linear_behaviour: eqx.nn.Linear
    key: int

    def __init__(self, N, M, O, key):
        super().__init__()
        self.linear_spikes = eqx.nn.Linear(N, M, key=key)
        self.linear_behaviour = eqx.nn.Linear(M, O, key=key)

        self.key = key
    
    def __call__(self, x):
        rates =  jnp.exp(self.linear_spikes(x))
        spikes = jrandom.poisson(self.key, rates)
        behaviour = self.linear_behaviour(rates)
        return rates, spikes, behaviour




class Rates_to_Obv(eqx.Module):
    rates_to_obv: eqx.nn.Linear

    def __init__(self, M, O, key):
        super().__init__()
        self.rates_to_obv = eqx.nn.Linear(M, O, key=key)
    
    def __call__(self, rates):
        obs = jax.vmap(self.rates_to_obv)(rates)
        return obs







def visualize_rates(ax, preds, gt):

    """ Function fo visulaizing trajectories """
    sample_n = 0
    # get_area_predictions and smoothed area_gt
    preds = preds[0] # gen_samples x time  x area_neurons
    gt = gt[0] #time  x area_neurons

    # get 10 neurons for plotting
    preds_10 =  jnp.expand_dims(np.moveaxis(preds,-1,0)[:10],-1)
    gt_10 = jnp.expand_dims(np.moveaxis(gt, -1, 0)[:10],-1)


    for i, (infered_rates, true_rates) in enumerate(zip(preds_10, gt_10)):
        # plot inferred rates
        ax[i].plot(infered_rates, label=f"Neuron {i+1}", c='g', linewidth=2)
        # plot true rates
        ax[i].plot(true_rates, label=f"Neuron {i+1} (GT)", linestyle='--', c= 'r', linewidth=2)
        ax[i].set_ylabel("Firing rate (Hz)")
        ax[i].legend(loc="upper right")
        #ax[i].set_ylim([-0.1, 1.1])

from scipy import stats

def visualize_rates_ci(ax, preds, gt):
    """ Function fo visulaizing trajectories  with sd"""
    sample_n = 0
    # get_area_predictions and smoothed area_gt
    N = preds.shape[0]
    T = preds.shape[2]
    preds = preds[:,0,:,:] # gen_samples x time  x area_neurons
    # preds_mean 
    preds_mean = jnp.mean(preds, axis=0)
    # preds_sd
    preds_sd = jnp.std(preds, axis=0)
    gt = gt[0] #time  x area_neurons

    # get 10 neurons for plotting
    preds_10_mean =  jnp.expand_dims(np.moveaxis(preds_mean,-1,0)[:10],-1)
    preds_10_sd =  jnp.expand_dims(np.moveaxis(preds_sd,-1,0)[:10],-1)
    gt_10 = jnp.expand_dims(np.moveaxis(gt, -1, 0)[:10],-1)


    for i, (infered_rates_m, infered_rates_sd, true_rates) in enumerate(zip(preds_10_mean, preds_10_sd, gt_10)):
        # plot inferred rates
        ax[i].plot(infered_rates_m, label=f"Neuron {i+1}", c='g', linewidth=2)
        # plot true rates
        ax[i].plot(true_rates, label=f"Neuron {i+1} (GT)", linestyle='--', c= 'r', linewidth=2)
        conf_interval = jnp.zeros((T, 2))
        for t in range(T):
            #percent_interval[t, :] = np.percentile(matrix[:, t], [2.5, 97.5])
            conf_interval[t, :] = stats.t.interval(0.95, N-1, loc=infered_rates_m[t], scale=infered_rates_sd[t]/np.sqrt(N))
            
        
        ax[i].fill_between(range(T), conf_interval[:,0], conf_interval[:,1], alpha=0.2)
        #ax[i].set_ylim([-0.5, 1.2])