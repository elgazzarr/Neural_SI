import argparse
import jax.numpy as jnp
import jax.numpy as jnp
from jax import lax
import jax.random as jrandom
import diffrax as dfx
import numpy as np
import wandb
import warnings
import yaml
from simulated_system.train import train_gt
from train import train
import pickle
from models import LatentODE, LatentSDE
from data import *
import os
from evaluate import eval
warnings.filterwarnings('ignore')


def main(args, config):
    # set random seed
    key = jrandom.PRNGKey(args.seed)
    # set wandb
    save_path = f'./results/{args.exp_name}/{args.run_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    wandb.init(project='Neural_DEs', name=args.run_name,  group=args.exp_name, config=config, dir=save_path, resume=True)
    config = wandb.config

    print('Experiment settings: ', config)
    
    print('Experiment name: {}'.format(args.exp_name))
    print('Run name: {}'.format(args.run_name))
    # set up the system
    if args.pretrained_gt_system is None:
        print('Training ground truth system from scratch')
        print('----------------------------------------')
        system = train_gt(key, config)
        print('Finished training ground truth system')
        print('Saving at: {}'.format(save_path))
        print('*'*50)
        #save the system
        with open(save_path + 'gt_system.pkl', 'wb') as f:
            pickle.dump(system, f) 


    else:
        print('Loading pretrained ground truth system from: {}'.format(args.pretrained_gt_system))
        print('----------------------------------------')
        system = pickle.load(open(args.pretrained_gt_system, 'rb'))


    keys = jrandom.split(key, 7)

    #Lets define our data and create our dataset loader
    dataset_size = config["data"]["dataset_size"]
    train_dataloader = Dataloader(system, dataset_size, config, keys[0])
    val_dataloader = Dataloader(system, dataset_size, config, keys[1])
    test_dataloader = Dataloader(system, dataset_size, config, keys[2])


    #Lets define our model
    if args.DE_type == 'ODE':
        print('Training an ODE-based model ...')
        model = LatentODE(key, config)
    elif args.DE_type == 'SDE':
        print('Training an SDE-based model ...')
        model = LatentSDE(key, config)
    else:
        raise NotImplementedError
    
    run_keys = jrandom.split(key, args.n_runs)
    for i in range(args.n_runs):
        print('Training run: {}'.format(i))
        trained_model = train((train_dataloader, val_dataloader), 
                            model, 
                            config, 
                            run_keys[i], 
                            args, 
                            save_path)
        print('Finished training model')
        print('----------------------------------------')
        
        # evaluate model
        print('Evaluating model ...')
        print('----------------------------------------')
        test_data_ll, test_acc = eval(trained_model, test_dataloader,  config, run_keys[i])
        wandb.log({'test_data_ll': round(test_data_ll,3)})
        print('Test data log-likelihood: {:.3f}'.format(test_data_ll))
        print('Test accuracy: {:.3f}'.format(test_acc))
        print('*'*50)
        wandb.log({'test_behaviour_acc': round(test_acc,3)})


        # save model
        if args.save_model:
            with open(save_path + 'model.pkl', 'wb') as f:
                pickle.dump(trained_model, f)
    # save files
    

    return 

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--run_name', type=str, default='trial1', help='run name')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--DE_type', type=str, default='SDE', choices=['ODE', 'SDE'], help='Type of underlying differential equation')
    parser.add_argument('--train_size', type=int, default=1024, help='Training dataset size')
    parser.add_argument('--process_noise', type=float, default=0.0, help='process noise')
    parser.add_argument('--pretrained_gt_system', type=str, default=None, help='Path to pretrained system acting as our gt, if none, train from scratch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--config', type=str, default='./configs/pilot.yaml', help='Path to config file')
    parser.add_argument('--save_model', type=bool, default=False, help='Save final model')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs (using different random seeds) to conduct the experiment')


    args = parser.parse_args()
    # load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config["data"]["dataset_size"] = args.train_size
    config["data"]["process_noise_scale"] = args.process_noise
    config['DE_type'] = args.DE_type

    main(args, config)



