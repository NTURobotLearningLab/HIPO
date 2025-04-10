config = {
    'device': 'cuda',
    'save_interval': 10,
    'log_interval': 10,
    'prefix': 'HighLevelPolicy',
    'algorithm': 'HighLevelPolicy',
    'mode': 'train',
    'seed': 123,
    'mdp_type': 'ProgramEnvHighLevelPolicy',
    'num_demo_per_program': 1,
    'height': 16,
    'width': 16,
    'env_task': 'infinite_harvester',
    'verbose': True,
    'env_name': 'karel',
    'gamma': 0.99,
    'max_episode_steps': 1000,
    'max_program_len': 45,
    'max_demo_length': 1000,
    'option_num': 5,
    'high_level_policy_checkpt': None,
    
    #For compatibility with LEAPS
    'AE': False,
    'two_head': False,
    'debug': False,
    'recurrent_policy': True,
    'num_lstm_cell_units': 256,
    'grammar':'handwritten',

    #For compatibility with LEAPS
    'net': {
        'saved_params_path': None,
        'rnn_type': 'GRU', 
        'tanh_after_mu_sigma': False, 
        'tanh_after_sample': False,
        'decoder': {
            'use_teacher_enforcing': True,
            'freeze_params': False,
        },

        'condition':{
            'use_teacher_enforcing': True,
            'observations': 'environment',
        },
        
        'controller':{
            'use_decoder_dist': True,
        },
    },

    #For compatibility with LEAPS
    'rl':{
        'value_method': 'mean',
        'envs': {
            'executable': {
                'max_demo_length': 1000,
                'num_demo_per_program': 10,
            },
        },
        
        'loss':{
            'condition_rl_loss_coef': 0.0,
            },

        'algo':{
            'name': 'reinforce',
        },
    },

    'logging': {
        'log_file': 'run.log',
        'fmt': '%(asctime)s: %(message)s',
        'level': 'DEBUG',
    },

    'dsl': {
        'use_simplified_dsl': False,                
        'max_program_len': 45,                   
        'grammar': 'handwritten',
    },
    
    'CEM':{
        'compatibility_vector': None,
    },
    
    # For high-level policy training
    'PPO':{
        'algo': 'ppo',
        'num_env_steps': 20e6,
        'num_steps': 32,
        'num_processes': 32,
        'use_linear_lr_decay': True,
        'decoder_deterministic': True,
        'use_gae': True,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'use_proper_time_limits': False,
        
        'hidden_size': 16,
        'lr': 5e-4,
        'eps': 1e-5,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'ppo_epoch': 4,
        'num_mini_batch': 128,
        'clip_param': 0.05,
        'val_interval': 128,
        'num_demo_val': 32,
    },
}
