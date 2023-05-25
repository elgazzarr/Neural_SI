import diffrax as dfx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax.numpy as jnp
import jax
import jax.random as jrandom
from metrics import poisson_nll
from utils import Rates_to_Obv
import optax
import equinox as eqx
from copy import deepcopy



def eval(model, data_loader, config, key):
    """Evaluate the data poissoin log likelihood"""
    ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])

    vis_keys = jrandom.split(key, 20)
    controls, spikes, rates, behaviour = data_loader.sample_observations(0)
    pred_rates = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, spikes, controls, jrandom.split(key, controls.shape[0]))[0] for key in vis_keys])
    pred_rates_mean = jnp.mean(pred_rates, axis=0)
    pred_behaviour = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, spikes, controls, jrandom.split(key, controls.shape[0]))[1] for key in vis_keys])
    pred_behaviour_mean = jnp.mean(pred_behaviour, axis=0)
    poisson_log_likelhood = poisson_nll(pred_rates_mean, spikes)
    acc = jnp.mean(jnp.argmax(pred_behaviour_mean, axis=-1) == jnp.argmax(behaviour, axis=-1))

    return poisson_log_likelhood, acc


def eval_behaviour(model, data_loaders, config, key):

    """Evaluate the R2 of rates for predicting behaviour using by training a linear layer to predict behaviour from rates"""

    ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])
    classifier = Rates_to_Obv(M=config['data']['n_neurons'], O=config['data']['behavior_size'], key=key)
    train_loader, val_loader, test_loader = data_loaders
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(classifier, eqx.is_array_like))

    # define the loss function
    best_loss = jnp.inf
    best_clf = None
    early_stop = 5
    es_counter = 0

    sample_keys = jrandom.split(key, 5)
    # Sample a batch of data for val 
    val_controls, val_spikes, _, val_behaviour = val_loader.sample_observations(0)
    val_rates = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, val_spikes, val_controls, jrandom.split(key, val_controls.shape[0]))for key in sample_keys])
    val_rates = jnp.mean(val_rates, axis=0)
    # Sample a batch of data for test
    test_controls, test_spikes, _, test_behaviour = test_loader.sample_observations(0)
    test_rates = jnp.array([jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, test_spikes, test_controls, jrandom.split(key, test_controls.shape[0]))for key in sample_keys])
    test_rates = jnp.mean(test_rates, axis=0)

    @eqx.filter_jit
    def loss_acc(classifier, rates, behaviour):
        pred_behvaiour = jax.vmap(classifier)(rates)
        loss = jnp.mean(optax.softmax_cross_entropy(pred_behvaiour, behaviour))
        acc = jnp.mean(jnp.argmax(pred_behvaiour, axis=-1) == jnp.argmax(behaviour, axis=-1))
        return loss, acc
    
    grad_loss = eqx.filter_value_and_grad(loss_acc, has_aux=True)

    @eqx.filter_jit
    def make_step(classifier, rates, behaviour, opt_state):
        (loss, acc), grads = grad_loss(classifier, rates, behaviour)
        updates, opt_state = optimizer.update(grads, opt_state)
        classifier = eqx.apply_updates(classifier, updates)
        return loss, acc, classifier, opt_state



    for step in range(200):
        controls, spikes, _, behaviour = train_loader.sample_observations(step)
        pred__rates = jax.vmap(model.predict, in_axes=[None, 0, 0, 0])(ts, spikes, controls, jrandom.split(key, controls.shape[0]))
        loss, acc, classifier, opt_state = make_step(classifier, pred__rates, behaviour, opt_state)
        if step % 20 == 0:
            val_loss, val_acc = loss_acc(classifier, val_rates, val_behaviour)
            print(f'Step: {step}, Train Loss: {loss:.2f}, Train Accuracy: {acc:.2f} || Val Loss: {val_loss:.2f}, Val Accuracy: {val_acc:.2f}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_clf = deepcopy(classifier)
                print('Saving best model')
                es_counter = 0
            else:
                es_counter += 1
            if es_counter == early_stop:
                print('Early stopping.. ')
                break
            
    _, test_acc = loss_acc(best_clf, test_rates, test_behaviour)

    return test_acc