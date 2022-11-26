import argparse
from SELFRec import SELFRec
import yaml
from yaml import SafeLoader
import wandb
from util.helper import fix_random_seed
from util.helper import composePath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace/')
    parser.add_argument('--dataset', type=str, default='iFashion')
    parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--config', type=str, default='default.yaml')
    parser.add_argument('--model', type=str, default='SGL')
    parser.add_argument('--tags', type=str)  # 这里是一些标签
    # Register your model here
    baseline = ['LightGCN','DirectAU','MF']
    graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF']
    sequential_models = []
    args = parser.parse_args()

    config_path = composePath(args.root, 'conf', args.model + '.yaml')
    config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset]
    wandb.init(project="gclrec",entity="oceanlvr",config={})
    wandb.config.update(args)
    wandb.config.update(config)
    fix_random_seed(wandb.config['seed'])

    if wandb.config['model'] not in graph_models:
          print("Wrong model name! Model {} is not supported".format(wandb.config['model']))
          exit(-1)
    rec = SELFRec(wandb.config)
    rec.execute()
