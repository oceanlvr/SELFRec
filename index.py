from SELFRec import SELFRec
import yaml
from yaml import SafeLoader
import wandb
from util.helper import fix_random_seed, composePath, ParseKwargs, mergeDict, splitNested
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SGL')
    parser.add_argument('--dataset', type=str, default='iFashion')
    parser.add_argument('--root', type=str, default='/workspace/')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--tags', nargs='*', default=[])  # 这里是一些标签
    parser.add_argument('--notes', type=str)
    parser.add_argument('--run_name', type=str)
    # graph args
    parser.add_argument('--ranking', nargs='*', type=int)
    parser.add_argument('--embbedding_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lambda', type=float)
    # model_config args
    parser.add_argument('-x', '-m', '--model_config', nargs='*',
                        action=ParseKwargs, help="-m num_layers=1 alpha=2", default={})
    # template SGL
    parser.add_argument('--model_config.droprate', type=float)
    parser.add_argument('--model_config.augtype', type=int)
    parser.add_argument('--model_config.temperature', type=float)
    parser.add_argument('--model_config.num_layers', type=int)
    parser.add_argument('--model_config.lambda', type=float)

    # template NCL
    parser.add_argument('--model_config.hyper_layers', type=int)
    parser.add_argument('--model_config.ssl_reg', type=float)
    parser.add_argument('--model_config.proto_reg', type=float)
    parser.add_argument('--model_config.alpha', type=float)
    parser.add_argument('--model_config.num_clusters', type=float)

    # Register your model here
    baseline = ['LightGCN', 'DirectAU', 'MF']
    graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR',
                    'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF']
    sequential_models = []
    args = parser.parse_args()
    args = splitNested(vars(args))

    config_path = composePath(args['root'], 'conf', args['model'] + '.yaml')
    config = yaml.load(open(config_path), Loader=SafeLoader)[args['dataset']]
    config = mergeDict(config, args)
    wandb.init(project="gclrec", entity="oceanlvr",
               name=args['run_name'] or None, config=config)

    fix_random_seed(wandb.config['seed'])
    print('='*10, 'wandb.config', '='*10)
    print(wandb.config)
    print('='*10, 'wandb.config', '='*10)

    if wandb.config['model'] not in graph_models:
        print("Wrong model name! Model {} is not supported".format(
            wandb.config['model']))
        exit(-1)
    rec = SELFRec(wandb.config)
    rec.execute()
