#!/bin/bash

# Run bashrc
source ~/.bashrc


# Change directory to the desired location
cd code/NDEs_sim

export exp_name='ODEsvsSDEs2'


# Run the Python script with the specified argument
#python main.py --exp_name=$exp_name  --run_name='ODE_0.0' --DE_type=ODE --process_noise=0.0 


#python main.py --exp_name=$exp_name  --run_name='ODE_0.5' --DE_type=ODE --process_noise=0.5 

#python main.py --exp_name=$exp_name  --run_name='ODE_1.0' --DE_type=ODE --process_noise=1.0 

python main.py --exp_name=$exp_name --run_name='ODE_2.0' --DE_type=ODE --process_noise=2.0 


python main.py --exp_name=$exp_name --run_name='ODE_5.0' --DE_type=ODE --process_noise=5.0 


# Run the Python script with the specified argument
#python main.py --exp_name=$exp_name  --run_name='SDE_0.0' --DE_type=SDE --process_noise=0.0 

#python main.py --exp_name=$exp_name  --run_name='SDE_0.5' --DE_type=SDE --process_noise=0.5

#python main.py --exp_name=$exp_name  --run_name='SDE_1.0' --DE_type=SDE --process_noise=1.0 

python main.py --exp_name=$exp_name  --run_name='SDE_2.0' --DE_type=SDE --process_noise=2.0 

python main.py --exp_name=$exp_name  --run_name='SDE_5.0' --DE_type=SDE --process_noise=5.0 



