from SELFRec import SELFRec
import yaml
from yaml import SafeLoader
import wandb
from util.helper import fix_random_seed, composePath, mergeDict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SGL')
    parser.add_argument('--dataset', type=str, default='iFashion')
    parser.add_argument('--root', type=str, default='/workspace/')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--tags', nargs='*', default=[])  # 这里是一些标签
    parser.add_argument('--group', type=str, default='default')  #
    parser.add_argument('--job_type', type=str, default='eval')  #
    parser.add_argument('--notes', type=str)
    parser.add_argument('--run_name', type=str)
    # graph args
    parser.add_argument('--ranking', nargs='*', type=int)
    parser.add_argument('--embedding_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lambda', type=float)
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

    # template SimGCL
    parser.add_argument('--model_config.eps', type=float)
    parser.add_argument('--model_config.tau', type=float)

    # Register your model here
    baseline = ['LightGCN', 'DirectAU', 'MF']
    graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'LightGCN',
                    'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF', 'SwAVGCL']
    sequential_models = []
    args = vars(parser.parse_args())

    config_path = composePath(args['root'], 'conf', args['model'] + '.yaml')
    config = yaml.load(open(config_path), Loader=SafeLoader)[args['dataset']]
    config = mergeDict(config, args)
    wandb.init(project="selfrec_new", group=args['group'], job_type=args['job_type'],
               entity="oceanlvr", name=args['run_name'] or None, config=config)
    wandb.run.log_code(".")

    if wandb.config['model'] is not 'SGL':
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
