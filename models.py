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
from utils import Encoder, Vectorfield_mlp, Readout, Vectorfield_mlp_posterior, Diffusion, Diffusion_diag, Vectorfield_mlp_prior
from sde_kl import sde_kl_divergence

class AbstractDE(eqx.Module):

    key: jrandom.PRNGKey  
    encoder: Callable
    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.Linear
    drift_vf: Callable
    readout: Callable
    hidden_size: int
    latent_size: int

    def __init__(self, key, config):
        super().__init__()
        self.key = key

        self.encoder = Encoder(config['data']['n_neurons']+1, 
                            config['model']['hidden_size'], 
                            config['model']['width_size'], 
                            config['model']['depth'], key=key)
        
        self.hidden_to_latent = eqx.nn.Linear(config['model']['hidden_size'],
                                            config['model']['latent_size'],
                                            key=key)
        
        self.latent_to_hidden = eqx.nn.Linear(config['model']['latent_size'],
                                            config['model']['hidden_size'],
                                            key=key)
        
        self.drift_vf = Vectorfield_mlp_prior(config['model']['latent_size'], 
                                config['data']['control_size'],
                                config['model']['width_size'], 
                                config['model']['depth'],
                                key)
        
        self.readout = Readout(config['model']['hidden_size'],
                            config['data']['n_neurons'], 
                            config['data']['behavior_size'],
                            key=key)
        

        self.hidden_size = config['model']['hidden_size']
        self.latent_size = config['model']['latent_size']

    def _encode(self, ts, ys, key, get_context=False):

        data = jnp.concatenate([ts[:, None], ys], axis=-1)
        fwd_context = self.encoder(ts, data[::-1], get_context)
        #bwd_context = self.encoder(ts, data[::-1], get_context)
        context = fwd_context[::-1] #jnp.concatenate([fwd_context, bwd_context], axis=-1)
        context_init = context[0]
        y0 = self.hidden_to_latent(context_init)

        return (context, y0) if get_context else  y0
    
    def _solve(self, ts, y0, u, context):
        raise NotImplementedError
    
    def __call__(self, t, y, u):
        raise NotImplementedError
    
    def predict(self, ts, ys, us, key):
        raise NotImplementedError
    



class LatentODE(AbstractDE):

    def __init__(self, key, config):
        super().__init__(key, config)

    def _solve(self, ts, y0, u, key, context=None):
        if u is not None:
            u = dfx.LinearInterpolation(ts=ts, ys=u)
        system = dfx.ODETerm(self.drift_vf)
        solver = dfx.Euler()
        t = len(ts)//5
        ts = ts.reshape(5, t)
        dt = (ts[0,1] - ts[0,0]) * 0.5
        intervals =  jnp.append(ts[0,0],ts[:,-1] + dt)
        interval_begins = intervals[:-1]
        interval_endings = intervals[1:]
        carry = y0

        def truncated_solve(y0, t):
            ta , tb, ts = t
            dt0 = (ts[1] - ts[0]) * 0.5
            sol = dfx.diffeqsolve(
                system,
                solver,
                ta,
                tb,
                dt0=dt0,
                y0=y0,
                saveat=dfx.SaveAt(ts=ts),
                #stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
                args=u
                )
            ys = sol.ys
            y0 = jax.lax.stop_gradient(sol.ys[-1])
            return y0, ys
        
        carry, ys = jax.lax.scan(truncated_solve, carry, (interval_begins, interval_endings, ts))
        ys = ys.reshape(ys.shape[0]*ys.shape[1],ys.shape[2])
        hidden = jax.vmap(self.latent_to_hidden)(ys)
        rates, _,  behaviour  = jax.vmap(self.readout)(hidden)
        return rates, behaviour

    def __call__(self, ts, ys, us,  key):
        y0 = self._encode(ts, ys, key)
        return self._solve(ts, y0, us, key)

    def predict(self, ts, ys, us, key):
        y0 = self._encode(ts, ys, key)
        return self._solve(ts, y0, us, key)
    

class LatentSDE(AbstractDE):
    drift_vf_posterior: Callable
    diffusion: Callable

    def __init__(self, key, config):
        super().__init__(key, config)
        self.drift_vf_posterior = Vectorfield_mlp_posterior(config['model']['latent_size'],
                                                             config['data']['control_size'], 
                                                             config['model']['hidden_size'], 
                                                             config['model']['width_size'],
                                                            config['model']['depth'], 
                                                             key=self.key)

        self.diffusion =  Diffusion_diag(config['model']['latent_size'], key=self.key)



    def _solve(self, ts, y0, u, key, context):
        if u is not None:
            u = dfx.LinearInterpolation(ts=ts, ys=u)
        context = dfx.LinearInterpolation(ts=ts, ys=context)

        solver = dfx.Euler()
        t = len(ts)//5
        ts = ts.reshape(5, t)
        dt = (ts[0,1] - ts[0,0])
        intervals =  jnp.append(ts[0,0],ts[:,-1] + dt)
        interval_begins = intervals[:-1]
        interval_endings = intervals[1:]

        bm = dfx.VirtualBrownianTree(ts[0,0], ts[-1,-1] + (0.5*dt*ts.shape[0]),
                                        tol=dt*0.5, 
                                        shape=jax.ShapedArray(shape=(self.latent_size,),
                                                            dtype=jnp.float32), 
                                        key=key)
        control_term = dfx.WeaklyDiagonalControlTerm(self.diffusion, bm)
        posterior_drift = dfx.ODETerm(self.drift_vf_posterior)
        prior_drift = dfx.ODETerm(self.drift_vf)

        #get our agumented sde
        aug_sde, aug_y0 = sde_kl_divergence(
             drift1=posterior_drift,
             drift2=prior_drift,
             diffusion=control_term,
             y0=y0)
        
        def truncated_solve(carry, t):
            y0 = carry
            ta , tb, ts = t
            dt0 = (ts[1] - ts[0]) * 0.5
            sol = dfx.diffeqsolve(aug_sde,
                                t0=ta,
                                t1=tb,
                                y0=y0,
                                dt0=dt0,
                                solver=solver,
                                #stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
                                saveat=dfx.SaveAt(ts=ts),
                                args=[context, u])
            
            ys = sol.ys # A tuple that contatins the solution for both SDEs
            
            y1 = jax.lax.stop_gradient(sol.ys[0][-1]) # The last time step solution for the posterior SDE
            
            y2 = jax.lax.stop_gradient(sol.ys[1][-1])

            carry = (y1,y2)

            return carry, ys

        carry = (aug_y0, 0.0)
        carry, ys = jax.lax.scan(truncated_solve, carry, (interval_begins, interval_endings, ts))
        zs = ys[0]
        logpq_path = ys[1]
        zs = zs.reshape(zs.shape[0]*zs.shape[1],zs.shape[2])
        logpq_path = logpq_path.reshape((-1,))
        hidden = jax.vmap(self.latent_to_hidden)(zs)
        rates, _,  behaviour  = jax.vmap(self.readout)(hidden)
        
        return rates, logpq_path, behaviour

    def __call__(self, ts, ys, us,  key):
        context, y0 = self._encode(ts, ys, key, get_context=True)
        pred_rates, logpq_path, behaviour = self._solve(ts, y0, us, key, context)
        logpq =  logpq_path[-1]
        return  pred_rates, logpq, behaviour
    
    def predict(self, ts, ys, us,  key):
        context, y0 = self._encode(ts, ys, key, get_context=True)
        pred_rates, logpq_path, pred_behaviour = self._solve(ts, y0, us, key, context)
        return  pred_rates, pred_behaviour

    def predict_prior(self, ts, ys, us, key):
        context, y0 = self._encode(ts, ys, key, get_context=True)
        u = dfx.LinearInterpolation(ts=ts, ys=us)
        dt = (ts[1] - ts[0]) * 0.5
        bm = dfx.VirtualBrownianTree(ts[0], ts[-1],
                                tol=dt*0.5, 
                                shape=jax.ShapedArray(shape=(self.latent_size,),
                                                    dtype=jnp.float32), 
                                key=key)
        drift = dfx.ODETerm(self.drift_vf)
        diffusion = dfx.WeaklyDiagonalControlTerm(self.diffusion, bm)
        system = dfx.MultiTerm(drift, diffusion)
        sol = dfx.diffeqsolve(system,
                    solver=dfx.ReversibleHeun(),
                    t0=ts[0],
                    t1=ts[-1],
                    y0=y0,
                    dt0=dt,
                    stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
                    #max_steps=16**4,
                    saveat=dfx.SaveAt(ts=ts),
                    args=u).ys
        
        hidden = jax.vmap(self.latent_to_hidden)(sol)
        pred_rates, _,  pred_behaviour  = jax.vmap(self.readout)(hidden)

        return pred_rates, pred_behaviour