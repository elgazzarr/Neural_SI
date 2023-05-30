import jax
import equinox as eqx
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jcp 
import optax
import math
from typing import Callable, List
import jax.nn as jnn
import jax.numpy as jnp
import ipdb
from jax import jit
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import os
from jax import Array
from diffrax.custom_types import PyTree, Scalar, Union
import warnings
from scipy.special import gammaln
from copy import deepcopy
from simulated_system.models import System_A, System_B, Readout
from simulated_system.data import Sys_Dataloader
warnings.filterwarnings("ignore")



class Static_Diff(eqx.Module):
    W: Array 
    def __init__(self, N, noise_scale):
        super().__init__()
        self.W = jnp.eye(N) * noise_scale
    
    def __call__(self, t, y, args):
        return self.W


class SystemDE(eqx.Module):
    f_vf: Callable
    f_obv: Callable
    key: int
    diffusion: Callable

    def __init__(self, f_vf, f_obv, f_diff, key):
        """
        params:
            f_vf: vector field. Here will be represented by our RNN. [N --> N x C]
            f_obv: readout function (Linear matrix).  [N --> M]
        """
        super().__init__()

        self.f_vf= f_vf
        self.f_obv = f_obv
        self.key = key
        self.diffusion = f_diff
    
    def __call__(self, ts, control, key):

        """
        params:
            ts: timesteps vector to create control  [T]
            phases: phases vector to create control [C]
            frequencies :  freqs vector to create control [C]
            init: inital state of the CDE []
        
        returns:
            observations of the CDE, which is the solution of the CDE passed through a observation function.
        """
        # add time as a channel to the control
        control = jnp.concatenate([ts[:,None], control], axis=-1)
        control = dfx.LinearInterpolation(ts, control)
        dt0 = (ts[1] - ts[0])/2
        drift = dfx.ODETerm(self.f_vf)
        bm = dfx.VirtualBrownianTree(t0=ts[0], t1=ts[-1], tol = dt0*0.5, 
                                     shape=jax.ShapedArray(shape=(self.f_vf.N,),
                                                            dtype=float),
                                                              key=key)
        diffusion = dfx.ControlTerm(jax.lax.stop_gradient(self.diffusion), bm)
        system = dfx.MultiTerm(drift, diffusion)
        solver = dfx.Euler()
        
        y0 = jnp.zeros(self.f_vf.N) 
        sol = dfx.diffeqsolve(system, solver, ts[0], ts[-1], dt0, y0=y0,
                            saveat=dfx.SaveAt(ts=ts),
                            #stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
                            args=control)    
        rates, spikes, obs = jax.vmap(self.f_obv)(sol.ys)
        
        return rates, spikes, obs



def train(model, dataloaders, ts, key):
    train_dataloader, val_dataloader = dataloaders
    # define the optimizer
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array_like))
    # define the loss function
    best_loss = jnp.inf
    best_model = None
    early_stop = 2
    es_counter = 0

    
    def loss_acc(model, controls, outputs):
        keys = jrandom.split(key, controls.shape[0])
        _, _, preds = jax.vmap(model, in_axes=[None, 0, 0])(ts, controls, keys)
        loss = jnp.mean(optax.softmax_cross_entropy(preds,outputs))
        # Compute the accuracy
        acc = jnp.mean(jnp.argmax(preds, axis=-1) == jnp.argmax(outputs, axis=-1))
        return loss, acc
    
    grad_loss = eqx.filter_value_and_grad(loss_acc, has_aux=True)

    @eqx.filter_jit
    def make_step(model, controls, outputs, opt_state):
        (loss, acc), grads = grad_loss(model, controls, outputs)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, acc, model, opt_state

    # training loop
    #print('Starting training of GT System')
    for step in range(1001):
        # sample a batch of observations
        controls, outputs = train_dataloader.sample_observations(step)
        #Make a step 
        loss, acc, model, opt_state = make_step(model, controls, outputs, opt_state)

        if step % 100 == 0:
            val_controls, val_outputs = val_dataloader.sample_observations(0)
            val_loss, val_acc = loss_acc(model, val_controls, val_outputs)
            print(f'Step: {step}, Train Loss: {loss:.2f}, Train Accuracy: {acc:.2f} || Val Loss: {val_loss:.2f}, Val Accuracy: {val_acc:.2f}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(model)
                print('Saving best model')
                es_counter = 0
            else:
                es_counter += 1
            if es_counter == early_stop:
                print('Early stopping.. ')
                break
            

    return best_model



def train_gt(key, config):
    keys = jrandom.split(key, 7)
    dataset_size = config['data']['dataset_size']
    train_dataloader = Sys_Dataloader(dataset_size, config, keys[0])
    val_dataloader = Sys_Dataloader(dataset_size//8, config, keys[1])
    # lets define our model
    N = config['data']['latent_size']
    O = config['data']['behavior_size']
    M = config['data']['n_neurons']
    C = config['data']['control_size']
    ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])

    # define the vector field using system_A 
    f_vf = System_A(N, C, keys[2])
    f_readout = Readout(N, M, O, keys[3])
    f_diff = Static_Diff(N, config['data']['process_noise_scale'])

    model = SystemDE(f_vf, f_readout, f_diff, key)
    trained_model = train(model, (train_dataloader, val_dataloader), ts, keys[4])
    return trained_model
