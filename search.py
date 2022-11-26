import wandb
import numpy as np 
import random

# Define sweep config
sweep_configuration = {
    'program': 'python index.py --gpu_id=0 --model=SGL --dataset=iFashion',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'NDCG'},
    'parameters': {
        'embbedding_size': {'values': [16, 32, 64]},
        'batch_size': {'values': [512, 1024, 2048]},
        'learning_rate': {'max': 0.1, 'min': 0.0001},
        'lambda': {'max': 0.01, 'min': 0.0001},
        'model_config.droprate': {'max': 0.99, 'min': 0.01},
        'model_config.augtype': {'values': [0,1]},
        'model_config.temperature': {'max': 0.9, 'min': 0.1},
        'model_config.num_layers': {'values': [1,2,3]},
        'model_config.lambda': {'max': 0.5, 'min': 0.05},
     },
     'project': "sweep_gclrec",
     'entity': "oceanlvr",
     'run_cap': 500,
}

if __name__ == '__main__':
  # Initialize sweep by passing in config. (Optional) Provide a name of the project.
  sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
  # Start sweep job.
  wandb.agent(sweep_id, count=4)
