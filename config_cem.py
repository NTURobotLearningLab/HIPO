config = {
    'device': 'cuda',
    'save_interval': 10,
    'log_interval': 10,
    'prefix': 'CEM',
    'algorithm': 'CEM',
    'mode': 'train',
    'seed': 123,
    'mdp_type': 'ProgramEnvCEM',
    'num_demo_per_program': 10,
    'height': 16,
    'width': 16,
    'env_task': 'infinite_harvester',
    'verbose': True,
    'env_name': 'karel',
    'gamma': 0.99,
    'max_episode_steps': 1000,
    'max_program_len': 45,
    'max_demo_length': 1000,
    
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
        'init_type': 'normal',
        'reduction': 'weighted_mean',
        'population_size': 32,
        'elitism_rate': 0.1,
        'max_number_of_epochs': 1000,
        'sigma': 0.5,
        'final_sigma': 0.1,
        'use_exp_sig_decay': True,
        'exponential_reward': False,

        'diversity_vector': None,
        'compatibility_vector': None,
        'gamma': 1,
        'random_seq': None,
        'vertical_similarity': True,
        'early_stop_std': 0.0001
    },
}
