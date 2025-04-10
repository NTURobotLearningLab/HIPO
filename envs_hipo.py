import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch

import gym
from gym.spaces.box import Box
from gym import spaces

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from leaps.rl.envs import VecNormalize, VecPyTorchFrameStack, VecPyTorch, TransposeImage

from leaps.karel_env.dsl import get_DSL
from leaps.karel_env.dsl.dsl_parse import parse

from leaps.prl_gym.exec_env import ExecEnv
from leaps.prl_gym.program_env import ProgramEnv

from karel_long_env.generator_karel_long import KarelLongStateGenerator
from karel_long_env.karel_long import KarelLong_world

class ExecEnvCEM(ExecEnv):
    def __init__(self, config, metadata={}):
        self.config = config
        self.dsl = get_DSL(dsl_type='prob', seed=config.seed, environment=self.config.env_name)
        self.s_gen = KarelLongStateGenerator(seed=config.seed)
        self._world = KarelLong_world(env_task=config.env_task)

        if config.env_task == 'infinite_harvester':
            self.init_func = self.s_gen.generate_single_state_infinite_harvester
        elif config.env_task == 'seesaw':
            self.init_func = self.s_gen.generate_single_state_seesaw
        elif config.env_task == 'upNdown':
            self.init_func = self.s_gen.generate_single_state_upNdown
        elif config.env_task == 'farmer':
            self.init_func = self.s_gen.generate_single_state_farmer
        elif config.env_task == 'infinite_doorkey':
            self.init_func = self.s_gen.generate_single_state_infinite_doorkey
        else:
            raise NotImplementedError('task not implemented yet')

        self.init_states = [self.init_func(config.height, config.width) for _ in range(config.num_demo_per_program)]
        self._world.set_new_state(self.init_states[0][0], self.init_states[0][4])

    def execute_pred_program(self, program_seq, compatibility_program, random_seqs):
        self._world.clear_history()
        h, w, c = self._world.s_h[0].shape
        a_h_list = []
        r_h_list = []
        fr_h_list = []
        pred_program = {}

        program_str = self.dsl.intseq2str(program_seq)
        exe, s_exe = parse(program_str, environment=self.config.env_name)

        program_count = 0
        if not s_exe or not len(program_seq) > 4:
            # can't execute the program or it's a dummy program: DEF run m()m
            syntax = False
            a_h_list.append(np.array(self._world.a_h).shape[0])
            r_h_list.append(-1)
            fr_h_list.append(-1)
            
        else:
            syntax = True
            for random_seq in random_seqs:
                for k in range(self.config.num_demo_per_program):
                    self._world.clear_history()
                    self._world.set_new_state(self.init_states[k][0], self.init_states[k][4])

                    for p in random_seq:
                        if p == "c":
                            exe, s_exe = parse(program_str, environment=self.config.env_name)
                        else:
                            exe, s_exe = parse(compatibility_program[int(p)], environment=self.config.env_name)

                        if not s_exe:
                            raise RuntimeError('This should be correct')

                        self._world, n, s_run = exe(self._world, 0)
                        program_count += 1

                    a_h_list.append(np.array(self._world.a_h).shape[0])
                    r_h_list.append(np.array(self._world.r_h).sum()/len(random_seq))
                    fr_h_list.append(np.array(self._world.r_h).sum())

        # save the state
        pred_program['a_h_len'] = np.array(a_h_list, dtype=np.int32)
        pred_program['program'] = program_seq
        pred_program['num_execution'] = program_count #self.config.num_demo_per_program
        pred_program['program_prediction'] = program_str
        pred_program['program_syntax'] = 'correct' if syntax else 'wrong'
        pred_program['mean_reward'] = np.mean(r_h_list)
        pred_program['mean_full_reward'] = np.mean(fr_h_list)
        pred_program['max_full_reward'] = np.max(fr_h_list)
        if pred_program['program_syntax'] == 'correct':
            pred_program['mean_reward'] += 0.1

        return pred_program

    def reward(self, pred_program_seq, compatibility_program, random_seq):
        pred_program = self.execute_pred_program(pred_program_seq, compatibility_program, random_seq)
        reward = pred_program['mean_reward']
        
        return reward, pred_program

    def reset(self):
        self.init_states = [self.init_func(self.config.height, self.config.width) for _ in range(self.config.num_demo_per_program)]
        return

class ExecEnvHighLevelPolicy(ExecEnv):
    def __init__(self, config, metadata={}):
        self.config = config
        self.dsl = get_DSL(dsl_type='prob', seed=config.seed, environment=self.config.env_name)
        self.s_gen = KarelLongStateGenerator(seed=config.seed)
        self._world = KarelLong_world(env_task=config.env_task)

        if config.env_task == 'infinite_harvester':
            self.init_func = self.s_gen.generate_single_state_infinite_harvester
        elif config.env_task == 'seesaw':
            self.init_func = self.s_gen.generate_single_state_seesaw
        elif config.env_task == 'upNdown':
            self.init_func = self.s_gen.generate_single_state_upNdown
        elif config.env_task == 'farmer':
            self.init_func = self.s_gen.generate_single_state_farmer
        elif config.env_task == 'infinite_doorkey':
            self.init_func = self.s_gen.generate_single_state_infinite_doorkey
        else:
            raise NotImplementedError('task not implemented yet')

        self.init_states = [self.init_func(config.height, config.width)]
        self._world.set_new_state(self.init_states[0][0], self.init_states[0][4])

    def execute_pred_program(self, program_seq):
        h, w, c = self._world.s_h[0].shape
        a_h = 0
        fr_h = 0
        pred_program = {}

        program_str = self.dsl.intseq2str(program_seq)
        exe, s_exe = parse(program_str, environment=self.config.env_name)

        if not s_exe or not len(program_seq) > 4:
            # can't execute the program or it's a dummy program: DEF run m()m
            syntax = False
            a_h = np.array(self._world.a_h).shape[0]
            fr_h = -1
        else:
            syntax = True
            exe, s_exe = parse(program_str, environment=self.config.env_name)
            if not s_exe:
                raise RuntimeError('This should be correct')

            self._world, n, s_run = exe(self._world, 0)

            a_h = np.array(self._world.a_h).shape[0]
            fr_h = np.array(self._world.r_h).sum()

        # save the state
        pred_program['current_state'] = self._world.s.copy()
        pred_program['done'] = self._world.done
        pred_program['a_h_len'] = a_h
        pred_program['program'] = program_seq
        pred_program['num_execution'] = 1
        pred_program['program_prediction'] = program_str
        pred_program['program_syntax'] = 'correct' if syntax else 'wrong'
        pred_program['reward'] = fr_h

        return pred_program

    def reward(self, pred_program_seq):
        pred_program = self.execute_pred_program(pred_program_seq)
        reward = pred_program['reward']
        
        return reward, pred_program

    def reset(self):
        self.init_states = [self.init_func(self.config.height, self.config.width)]
        self._world.set_new_state(self.init_states[0][0], self.init_states[0][4])
        return
        
class ProgramEnvCEM(ProgramEnv):
    def __init__(self, config, compatibility_program=[], random_seqs=[], metadata={}):
        self.config = config
        self.gt_reward = 1.1
        self.task_env = ExecEnvCEM(config, metadata)
        self.compatibility_program = compatibility_program
        self.random_seqs = random_seqs
        self.num_program_tokens = len(self.task_env.dsl.int2token)+1
        self.max_program_len = config.max_program_len

        self.action_space = spaces.Box(low=0, high=self.num_program_tokens, shape=(self.max_program_len,), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=self.num_program_tokens, shape=(self.max_program_len,), dtype=np.int8)
        self.initial_obv = (self.num_program_tokens-1) * np.ones(self.max_program_len, dtype=np.int8)
        self.state = self.initial_obv

    def _modify(self, action):
        null_token_idx = np.argwhere(action == (self.num_program_tokens-1))
        if null_token_idx.shape[0] > 0:
            action = action[:null_token_idx[0].squeeze()]
        action = self._prl_to_dsl(action) if self.config.use_simplified_dsl else action
        
        return action

    def step(self, action):
        self._elapsed_steps += 1
        dsl_action = self._modify(action)
        dsl_action = np.concatenate((np.array([0]), dsl_action))
        self.state = program_seq = dsl_action
        reward, exec_data = self.task_env.reward(program_seq, self.compatibility_program, self.random_seqs)
        done = True if reward == self.gt_reward else False
        info = {'cur_state': action, 'modified_action': dsl_action, 'exec_data': exec_data}

        done, info = self._set_bad_transition(done, info)

        return self.initial_obv, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        self.partial_program = []
        self.state = self.initial_obv
        self.task_env.reset()
        
        return self.state

class ProgramEnvHighLevelPolicy(ProgramEnv):
    def __init__(self, config, metadata={}):
        self.config = config
        self.gt_reward = 1.1
        self.task_env = ExecEnvHighLevelPolicy(config, metadata)
        self.num_program_tokens = len(self.task_env.dsl.int2token)+1
        self.max_program_len = config.max_program_len

        self.action_space = spaces.Box(low=0, high=self.num_program_tokens, shape=(self.max_program_len,), dtype=np.int8)
        self.initial_obv = self.task_env.init_states[0][0].copy()
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.initial_obv.shape), dtype=np.uint8)    
        self.state = self.initial_obv

    def _modify(self, action):
        null_token_idx = np.argwhere(action == (self.num_program_tokens-1))
        if null_token_idx.shape[0] > 0:
            action = action[:null_token_idx[0].squeeze()]
        action = self._prl_to_dsl(action) if self.config.use_simplified_dsl else action
        
        return action

    def step(self, action):
        self._elapsed_steps += 1
        dsl_action = self._modify(action[1:])
        dsl_action = np.concatenate((np.array([0]), dsl_action))
        self.state = program_seq = dsl_action
        reward, exec_data = self.task_env.reward(program_seq)
        
        self.state = exec_data['current_state']
        info = {'cur_state': action, 'modified_action': dsl_action, 'exec_data': exec_data}
        done = False
        done, info = self._set_bad_transition(done, info)

        if action[0] == self.config.option_num:
            done = True

        if done:
            self.state = self.reset()

        return self.state, reward, done, info
        
    def reset(self):
        self._elapsed_steps = 0
        self.partial_program = []
        self.state = self.initial_obv
        self.task_env.reset()
        
        return self.state
    
def make_env(env_id, seed, rank, log_dir, allow_early_resets, custom_env=False, custom_env_type='program',
             custom_kwargs=None):
    def _custom_program_thunk():
        # create env
        assert custom_kwargs is not None, 'Need task definition and configs for environment'
        if custom_kwargs['config'].mdp_type == 'ProgramEnvCEM':
            env = ProgramEnvCEM(**custom_kwargs)
        elif custom_kwargs['config'].mdp_type == 'ProgramEnvHighLevelPolicy':
            env = ProgramEnvHighLevelPolicy(**custom_kwargs)
        else:
            raise NotImplementedError()
        env._max_episode_steps = custom_kwargs['config'].max_episode_steps

        # set seed
        env.seed(seed + rank)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    if custom_env:
        if custom_env_type == 'program':
            return _custom_program_thunk
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    
def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  custom_env=False,
                  custom_env_type='program',
                  custom_kwargs=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, custom_env, custom_env_type, custom_kwargs)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1 and not custom_env:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        pass

    return envs