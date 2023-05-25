from simulated_system.controller import Controller
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx
import diffrax as dfx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax

def generate_observation(ts, t_go, t_stim, phases, frequencies, init):

    controller = Controller(t_stim, phases, frequencies, frequencies.shape[0])
    output = np.zeros_like(ts)
    f = lambda t, x, args: controller.evaluate(t)
    y0 = init
    control_signal = dfx.diffeqsolve(terms=dfx.ODETerm(f), solver= dfx.ReversibleHeun(),
                            t0=ts[0], t1=ts[-1], y0=y0, dt0=(ts[1] - ts[0])/2, saveat=dfx.SaveAt(ts=ts)).ys
    
    # Output is 0 until t_go, then it is 1 if sum of the control signal is greater than 0, and -1 otherwise
    control_signal_sum = jnp.sum(control_signal, axis=1)
    output_singal = jnp.where(control_signal_sum > 0, 1, -1)
    output = jnp.where(ts> t_go, output_singal, 0)


    return  control_signal, output


### DataLoader

class Sys_Dataloader():
    def __init__(self, dataset_size, config, key):

        batch_size = config['data']['batch_size']
        ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])
        t_stim = config['data']['t_stim']
        t_wait = config['data']['t_wait']
        C = config['data']['control_size']
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        keys = jrandom.split(key,5)
        self.freqs = jrandom.uniform(keys[0], shape=(self.dataset_size, C), minval=0.5, maxval=2)
        self.phases = jrandom.uniform(keys[1], shape=(self.dataset_size, C))
        self.t_stim = jrandom.uniform(keys[2], shape=(self.dataset_size,), minval=t_stim, maxval=t_stim)
        self.t_go = self.t_stim + t_wait
        init =  jrandom.normal(keys[3], shape=(self.dataset_size, C))
        self.controls, self.outputs = jax.vmap(generate_observation, in_axes=(None,0,0,0,0,0))(ts, self.t_go, self.t_stim,
                                                                                self.phases, self.freqs, init)
        # One hot encode the output
        self.outputs = jnn.one_hot(self.outputs+1, num_classes=config['data']['control_size'])
        self.sample_key = keys[5]
        

    def sample_observations(self, epoch):
        # Returns  control input to both GT and our model (input freqs and phases) and the observations of the ground truth system (Singal used for training)
        new_key = jrandom.fold_in(self.sample_key, epoch)
        indicies = jrandom.randint(new_key, shape=(self.batch_size,), minval=0, maxval=self.dataset_size)
        

        return self.controls[indicies], self.outputs[indicies]


def plotting(ts, controls, gts, t_stim, t_go):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    ax1.plot(ts, controls[1,:,:])
    # add vertical lines to indicate the start and end of the stimulus
    ax1.axvline(t_stim, color='r', linestyle='--', label='Stimulus Start')
    ax1.axvline(t_go, color='g', linestyle='--', label='Go Signal')

    ax2.axvline(t_stim, color='r', linestyle='--', label='Stimulus Start')
    ax2.axvline(t_go, color='g', linestyle='--', label='Go Signal')
    # add a title to the first subplot
    ax1.set_title('Control Signal')
    # add a title to the second subplot
    ax2.set_title('Output Signal')
    # add a legend to the first subplot
    ax1.legend()
    # add shades to indicate the time when the stimulus is on
    ax2.axvspan(t_stim, t_go, alpha=0.1, color='r', label='Fixation')
    ax2.legend()
    ax2.plot(ts, jnp.argmax(gts[1],axis=-1)-1)
    plt.show()
    # Save plot to file
#plt.savefig('system_input-output.png', dpi=300)