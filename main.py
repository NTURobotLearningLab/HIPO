import sys
import time
import os
import random
import logging
import numpy as np
import pickle
import shutil
import torch
from tensorboardX import SummaryWriter
sys.path.insert(0, 'HPRL')
sys.path.insert(0, 'karel_long_env')
sys.path.insert(0, 'leaps/karel_env')
sys.path.insert(0, 'leaps/karel_env/dsl')
sys.path.insert(0, 'leaps/rl')
sys.path.insert(0, 'leaps/prl_gym')
sys.path.insert(0, 'leaps')

from leaps.karel_env.dsl import get_DSL
from leaps.fetch_mapping import fetch_mapping
from leaps.pretrain import customargparse
from leaps.rl import utils
from leaps.rl.envs import make_vec_envs
from leaps.pretrain.misc_utils import create_directory
from envs_hipo import make_vec_envs
from CEM_hipo import CEMModelHIPO

from high_level_policy_hipo import HighLevelPolicy

def run(config, logger):
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
    else:
        raise NotImplementedError("No GPUs found")

    logger.debug('{} Using device: {}'.format(__name__, device))

    writer = SummaryWriter(logdir=config['outdir'])

    logger.debug('{} Setting random seed'.format(__name__))
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    global_logs = {'info': {}, 'result': {}}
    custom_kwargs = {"config": config['args']}
    custom = True
    
    envs = make_vec_envs(config['env_name'], config['seed'], 1,
                         config['gamma'], os.path.join(config['outdir'], 'env'), device, False, custom_env=custom,
                         custom_kwargs=custom_kwargs)

    dsl = get_DSL(seed=seed, environment=config['env_name'])
    config['dsl']
    config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1 # +1 for a no-op action, just for filling
    if config['algorithm'] == 'CEM':
        model = CEMModelHIPO(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    elif config['algorithm'] == 'HighLevelPolicy':
        model = HighLevelPolicy(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    else:
        raise NotImplementedError()

    # Save configs and models
    pickle.dump(config, file=open(os.path.join(config['outdir'], 'config.pkl'), 'wb'))
    shutil.copy(src=config['configfile'], dst=os.path.join(config['outdir'], 'configfile.py'))
    
    if config['algorithm'] == 'CEM':
        model.train()
    elif config['algorithm'] == 'HighLevelPolicy':
        model.train()
    else:
        raise NotImplementedError()
        
    return
    
if __name__ == "__main__":
    t_init = time.time()
    parser = customargparse.CustomArgumentParser(description='syntax learner')
    parser.add_argument('-o', '--outdir', default='output_dir')
    parser.add_argument('-c', '--configfile')

    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, '%s-%s-%s' % (args.prefix, args.seed, time.strftime("%Y%m%d-%H%M%S")))
    
    _, _, args.dsl_tokens, _ = fetch_mapping('leaps/mapping_karel2prl.txt')
    args.use_simplified_dsl = False

    config = customargparse.args_to_dict(args)
    config['args'] = args

    print(config['outdir'])
    create_directory(config['outdir'])

    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    run(config, logger)
