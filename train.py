import jax
import equinox as eqx
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jcp 
import optax
from typing import Callable, List
import jax.nn as jnn
import jax.numpy as jnp
import ipdb
from jax import jit
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.special import gammaln
from copy import deepcopy
warnings.filterwarnings("ignore")
import time 
import pickle
from metrics import *
from utils import *
from models import *
import wandb




def train(dataloaders, model, config, key, args, save_path):

    ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])
    keys = jrandom.split(key, 10)
    train_dataloader, val_dataloader = dataloaders

    steps = config['train']['steps']
    optim = optax.adam(config['train']['lr'])
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    val_controls, val_spikes, val_rates, val_behaviour = val_dataloader.sample_observations(0)
    best_val_loss = jnp.inf
    best_model = None
    es_counter = 0
    early_stop = config['train']['early_stopping']


    @eqx.filter_value_and_grad
    def ode_loss(model, ts, ys_n, ys_b, us, key, l ):
        batch_size = ys_n.shape[0]
        key = jrandom.split(key, batch_size)
        pred_rates, pred_behaviour  = jax.vmap(model, in_axes=[None, 0, 0, 0])(ts, ys_n, us, key)
        poisson_log_likelhood = poisson_nll(pred_rates, ys_n)
        behaviour_ce = jnp.mean(optax.softmax_cross_entropy(pred_behaviour, ys_b))
 
        jax.debug.print('pll: {}', poisson_log_likelhood)
        jax.debug.print('ce: {}', behaviour_ce)
        #wandb.log({'pll': poisson_log_likelhood, 'ce': behaviour_ce})
    
        loss = 0.7*poisson_log_likelhood + 0.3*behaviour_ce
        return loss


    @eqx.filter_value_and_grad
    def sde_loss(model, ts, ys_n, ys_b, us, key, l ):
        batch_size = ys_n.shape[0]
        key = jrandom.split(key, batch_size)
        pred_rates, logpq, pred_behaviour = jax.vmap(model, in_axes=[None, 0, 0, 0])(ts, ys_n, us, key)
        poisson_log_likelhood = poisson_nll(pred_rates, ys_n)
        behaviour_ce = jnp.mean(optax.softmax_cross_entropy(pred_behaviour, ys_b))
        kl_process = jnp.mean(logpq)
        jax.debug.print('pll: {}', poisson_log_likelhood)
        jax.debug.print('ce: {}', behaviour_ce)
        jax.debug.print('kl_process: {}', kl_process)
        #wandb.log({'pll': poisson_log_likelhood, 'ce': behaviour_ce, 'kl_process': kl_process})
        loss = poisson_log_likelhood + 0.3*behaviour_ce +  l*kl_process 

        return loss


    loss = ode_loss if args.DE_type == 'ODE' else sde_loss

    @eqx.filter_jit
    def make_step(model, opt_state, ts, ys_n, ys_b, us, key, l):
        value, grads = loss(model, ts, ys_n, ys_b, us, key, l)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        return value, model, opt_state

    vis_samples = 20 if args.DE_type == 'SDE' else 1 
    vis_keys = jrandom.split(keys[0], vis_samples) 
    start = time.time()
    #Sample control
    for e in range(steps):
        
        controls, spikes, _, behaviour  = train_dataloader.sample_observations(e)
        lambda_kl = jnp.minimum(e/config['train']['kl_annealing_steps'], 1.0)
        value, model, opt_state = make_step(model, opt_state, ts, spikes, behaviour, controls, keys[7], jnp.array(lambda_kl))
        value = value.item()
        if (e)%config['train']['print_every']  == 0:
            end = time.time()
            print(r"Step: {}, The loss is: {:.3f}, time: {:.3f} s".format(e, value, end-start))
            pred_val_rates = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, val_spikes, val_controls, jrandom.split(key, val_controls.shape[0]))[0] for key in vis_keys])
            pred_val_behaviour = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, val_spikes, val_controls, jrandom.split(key, val_controls.shape[0]))[1] for key in vis_keys])

            pred_val_rates_mean = jnp.mean(pred_val_rates, axis=0)
            pred_val_behaviour_mean = jnp.mean(pred_val_behaviour, axis=0)
            val_poisson_log_likelhood = poisson_nll(pred_val_rates_mean, val_spikes)
            val_ce_behaviour = jnp.mean(optax.softmax_cross_entropy(pred_val_behaviour_mean, val_behaviour))
            val_loss = val_poisson_log_likelhood + val_ce_behaviour
            val_acc = jnp.mean(jnp.argmax(pred_val_behaviour_mean, axis=-1) == jnp.argmax(val_behaviour, axis=-1))

            print(r"The validation loss is: {:.3f}".format(val_loss))


            wandb.log({'train_loss': value, 'val_pll': val_poisson_log_likelhood, 'val_acc': val_acc, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('saving best model.')
                wandb.log({'best_val_loss': best_val_loss})
                best_model = deepcopy(model)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > early_stop:
                    print('Early stopping at step {}'.format(e))
                    break
            fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(16, 16), sharex=True)
            visualize_rates(axes, pred_val_rates_mean, val_rates)
            plt.savefig(save_path + f'{e}.png')
            start = time.time()


    return best_model


