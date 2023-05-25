import diffrax as dfx
from simulated_system.controller import Controller
import jax
import jax.numpy as jnp
import jax.random as jrandom






def generate_observation(model, ts, t_stim, phases, frequencies, init, key):

    controller = Controller(t_stim, phases, frequencies, frequencies.shape[0])
    f = lambda t, x, args: controller.evaluate(t)
    y0 = init
    control_signal = dfx.diffeqsolve(terms=dfx.ODETerm(f), solver= dfx.Euler(),
                            t0=ts[0], t1=ts[-1], y0=y0, dt0=(ts[1] - ts[0])/2,
                            saveat=dfx.SaveAt(ts=ts)).ys

    rates, spikes, preds = model(ts, control_signal, key)



    return  control_signal, rates, spikes, preds


### DataLoader
class Dataloader():
    def __init__(self, model, dataset_size, config, key):

        batch_size = config['data']['batch_size']
        ts = jnp.linspace(0, config['data']['trial_time'], config['data']['n_timepoints'])
        t_stim = config['data']['t_stim']
        t_wait = config['data']['t_wait']
        C = config['data']['control_size']

        assert dataset_size%batch_size ==0
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        keys = jrandom.split(key,5)
        data_keys = jrandom.split(keys[4], dataset_size)
        self.freqs = jrandom.uniform(keys[0], shape=(self.dataset_size, C), minval=0.8, maxval=1.5)
        self.phases = jrandom.uniform(keys[1], shape=(self.dataset_size, C), minval=0, maxval=2*jnp.pi)
        self.t_stim = jrandom.uniform(keys[2], shape=(self.dataset_size,), minval=t_stim, maxval=t_stim)
        self.t_go = self.t_stim + t_wait
        init =  jrandom.uniform(keys[3], shape=(self.dataset_size, C))
        self.controls, self.rates, self.spikes, self.outputs = jax.vmap(generate_observation, in_axes=(None, None,0,0,0,0,0))(model, ts, self.t_stim,
                                                                                self.phases, self.freqs, init, data_keys)
        # calculate argmax of the output signal and one hot encode it
        self.outputs = jax.nn.one_hot(jnp.argmax(self.outputs, axis=-1), num_classes=3)
        self.sample_key = keys[5]
        

    def sample_observations(self, epoch):
        # Returns  control input to both GT and our model (input freqs and phases) and the observations of the ground truth system (Singal used for training)
        new_key = jrandom.fold_in(self.sample_key, epoch)
        indicies = jrandom.randint(new_key, shape=(self.batch_size,), minval=0, maxval=self.dataset_size)
        return self.controls[indicies], self.spikes[indicies], self.rates[indicies],  self.outputs[indicies]