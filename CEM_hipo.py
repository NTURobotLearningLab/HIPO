import os
import pickle
from collections import OrderedDict, deque
from operator import itemgetter
from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from leaps.pretrain.CEM import CrossEntropyNet, CrossEntropyAgent, CEMModel
from envs_hipo import make_vec_envs

class CrossEntropyAgentHIPO(CrossEntropyAgent):
    def __init__(self, device, logger, config, envs):
        super(CrossEntropyAgentHIPO, self).__init__(device, logger, config, envs)
        
        self.current_vector = self.model.get_init_vector(config['num_lstm_cell_units'], device)
        self._best_actual_score = 0
        self._best_mean_full_reward = 0
        self._best_max_full_reward = 0
        self._program_count = 0
        self._action_count = 0

        if config['CEM']['diversity_vector'] is not None and config['CEM']['diversity_vector'] != "None":
            diversity_vector = config['CEM']['diversity_vector']
            with open(diversity_vector, "rb") as input_file:
                diversity_vector = pickle.load(input_file)

                if isinstance(diversity_vector, dict):
                    self.diversity_vector = [diversity_vector]

                elif isinstance(diversity_vector, list):
                    self.diversity_vector = diversity_vector
                    
                else:
                    raise NotImplementedError()

        else:
            self.diversity_vector = None
            
        self.sigmoid = nn.Sigmoid()
        self.gamma = config['CEM']['gamma']

    @staticmethod
    def similarity(v1, v2):
        with torch.no_grad():
            similarity = F.cosine_similarity(v1, v2, dim=0)
        return similarity
        
    def scale(self, vector_i):
        similarity = Parallel(n_jobs=8)(delayed(self.similarity)(dv['vector'], vector_i) for dv in self.diversity_vector)
        max_similarity = torch.stack(similarity).max()
        with torch.no_grad():
            s = self.sigmoid(-max_similarity*self.gamma).cpu().item()

        return s

    def learn(self, envs, best_env):
        results = {}
        current_population = [self.current_vector + (self.current_sigma * torch.randn_like(self.current_vector)) for
                              _ in range(self.config['CEM']['population_size'])]

        current_population = torch.stack(current_population, dim=0)
        with torch.no_grad():
            pred_programs = self.act(current_population)

        obs, reward, done, infos = envs.step(pred_programs)

        if self.config['CEM']['exponential_reward']:
            reward = torch.exp(reward)
        
        for i, info in enumerate(infos):
            if self.diversity_vector is not None:
                vector_i = current_population[i]
                s = self.scale(vector_i)
                reward[i] = torch.mul(reward[i], s)
                
            results[i] = (reward[i].squeeze().detach().cpu().numpy(), info['exec_data']['program_prediction'])

            self._program_count += info['exec_data']['num_execution']
            self._action_count += info['exec_data']['a_h_len'].sum()

        sorted_results = OrderedDict(sorted(results.items(), key=itemgetter(1)))
        elite_idxs = list(sorted_results.keys())[-self.n_elite:]

        if self.reduction == 'mean':
            best_vector = torch.mean(current_population[elite_idxs], dim=0)
        elif self.reduction == 'max':
            best_vector, _ = torch.max(current_population[elite_idxs], dim=0)
        elif self.reduction == 'weighted_mean':
            reward = reward.to(self.device)
            best_vector = torch.sum(reward[elite_idxs] * current_population[elite_idxs], dim=0) / (torch.sum(reward[elite_idxs]) + 1e-5)
        with torch.no_grad():
            best_program = self.act(torch.stack((best_vector, best_vector)), deterministic=True)[0]

        _, best_reward, _, best_infos  = best_env.step(best_program.unsqueeze(0))
        if self.config['CEM']['exponential_reward']:
            best_reward = torch.exp(best_reward)
            
        best_actual_reward = best_reward.clone().detach().cpu().numpy()
        if self.diversity_vector is not None:
            s = self.scale(best_vector)
            best_reward = torch.mul(best_reward, s)
        best_reward = best_reward.detach().cpu().numpy()
        
        self.current_vector = best_vector

        if best_reward[0][0] > self._best_score:
            self._best_vector = best_vector
            self._best_program = best_program
            self._best_score = best_reward[0][0]
            self._best_actual_score = best_actual_reward[0][0]
            self._best_mean_full_reward = info['exec_data']['mean_full_reward']
            self._best_max_full_reward = info['exec_data']['max_full_reward']
            self._best_program_str = best_infos[0]['exec_data']['program_prediction']
        
        return results, pred_programs[elite_idxs].detach().cpu().numpy(),\
               current_population[elite_idxs].detach().cpu().numpy(), best_reward, best_actual_reward

    @property
    def best_actual_score(self):
        return self._best_actual_score
        
    @property
    def best_mean_full_reward(self):
        return self._best_mean_full_reward

    @property
    def best_max_full_reward(self):
        return self._best_max_full_reward

    @property
    def program_count(self):
        return self._program_count

    @property
    def action_count(self):
        return self._action_count
        
class CEMModelHIPO(CEMModel):
    def __init__(self, device, config, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = dummy_envs
        self.dsl = dsl
        
        if (config['CEM']['compatibility_vector'] is not None) and (config['CEM']['compatibility_vector'] != "None"):
            compatibility_vector = config['CEM']['compatibility_vector']
            with open(compatibility_vector, "rb") as input_file:
                compatibility_vector = pickle.load(input_file)

                if isinstance(compatibility_vector, dict):
                    self.compatibility_vector = [compatibility_vector]

                elif isinstance(compatibility_vector, list):
                    self.compatibility_vector = compatibility_vector
                    
                else:
                    raise NotImplementedError()
                    
            self.compatibility_vector = self.compatibility_vector[0:1]
            if "compatibility_vector" in self.compatibility_vector[0].keys():
                temp = self.compatibility_vector[0]["compatibility_vector"]
                if isinstance(temp, dict):
                    self.compatibility_vector.append(temp)

                elif isinstance(temp, list):
                    self.compatibility_vector.extend(temp)

        else:
            self.compatibility_vector = None
        
        self.compatibility_program = []
        if self.compatibility_vector is not None:
            for i in self.compatibility_vector:
                self.compatibility_program.append(i["program_str"])
                
        self.compatibility_program.reverse()

        if config["CEM"]['random_seq'] is not None:
            random_seq = config["CEM"]['random_seq']
            with open(random_seq) as file:
                random_seqs = [line.rstrip().split(",") for line in file]

            self.logger.debug("\nrandom_seqs: {}".format(random_seqs))

        else:
            raise NotImplementedError()

        custom = True
        logger.info('Using environment: {}, {}'.format(config["env_name"], config["env_task"]))
        custom_kwargs = {'config': config['args']}
        if config["mdp_type"] == "ProgramEnvCEM":
            custom_kwargs['compatibility_program'] = self.compatibility_program
            custom_kwargs['random_seqs'] = random_seqs
        else:
            raise NotImplementedError()
            
        self.envs = make_vec_envs(config["env_name"], config['seed'], config['CEM']['population_size'],
                                  config['gamma'], os.path.join(config['outdir'], 'CEM', 'env'), device, False,
                                  custom_env=custom, custom_kwargs=custom_kwargs)
        obs = self.envs.reset()

        self.best_env = make_vec_envs(config["env_name"], config['seed'], 1, config['gamma'],
                                      os.path.join(config['outdir'], 'CEM', 'env', 'best'), device, False,
                                      custom_env=custom, custom_kwargs=custom_kwargs)
        self.best_env.reset()


        self.agent = CrossEntropyAgentHIPO(device, logger, config, self.envs)

        if self.compatibility_vector is not None and config["CEM"]["vertical_similarity"]:
            try:
                self.agent.diversity_vector.extend(self.compatibility_vector)
            except:
                self.agent.diversity_vector = self.compatibility_vector

    def train(self):
        """run max_number_of_episodes learning epochs"""
        scores_deque = deque(maxlen=10)
        scores_deque_actual = deque(maxlen=100)
        scores = []
        for epoch in range(1, self.config['CEM']['max_number_of_epochs'] + 1):
            results, elite_programs, elite_vectors, reward, actual_reward = self.agent.learn(self.envs, self.best_env)
            self.agent.current_sigma = self.agent.sigma_sched.step(
                epoch - 1) if self.agent.sigma_sched.cur_step <= self.agent.sigma_sched.total_num_epoch else self.agent.final_sigma
            
            scores.append(reward)
            scores_deque.append(reward)
            scores_deque_actual.append(actual_reward)
            
            print_str = "Episode: {} - current mean Reward: {} best reward: {} best actual reward: {} best program: {} ".format(epoch, np.mean(scores_deque), self.agent.best_score, self.agent.best_actual_score, self.agent.best_program_str)
            
            self.logger.debug(print_str)
            
            # Early stop
            if np.std(scores_deque_actual) <= self.config["CEM"]["early_stop_std"] and len(scores_deque_actual) >= 100:
                self.logger.debug("\nEnvironment early stop after episode: {}".format(epoch))
                self.logger.debug("\nMean Reward over {} episodes: {}".format(epoch, np.mean(scores_deque)))
                self.save(os.path.join(self.config["outdir"], 'final_vectors.pkl'), converged=True)

                break
                
            if epoch == self.config['CEM']['max_number_of_epochs']:
                self.save(os.path.join(self.config["outdir"], 'final_vectors.pkl'), converged=False)

            if epoch % self.config['save_interval'] == 0:
                self.save(os.path.join(self.config["outdir"], 'final_vectors.pkl'))

            # Add logs to TB
            self.writer.add_scalar('agent/best_score', self.agent.best_score, epoch)
            self.writer.add_scalar('agent/best_actual_score', self.agent.best_actual_score, epoch)
            self.writer.add_scalar('agent/best_mean_full_reward', self.agent.best_mean_full_reward, epoch)
            self.writer.add_scalar('agent/best_max_full_reward', self.agent.best_max_full_reward, epoch)
            self.writer.add_scalar('agent/program_count', self.agent.program_count, epoch)
            self.writer.add_scalar('agent/action_count', self.agent.action_count, epoch)
            self.writer.add_text('program/best_{}'.format(epoch),
                                 'reward: ({}, {}) program: {} '.format(self.agent.best_score, self.agent.best_actual_score, self.agent.best_program_str), epoch)
            
                
        return scores

    def save(self, filename, converged=False):
        with open(filename, 'wb') as f:
            best_program = {'vector': self.agent.best_vector, 'program_str': self.agent.best_program_str,
                            'reward': self.agent.best_score, 'actual_reward': self.agent.best_actual_score, 'converged': converged}
                            
            if self.compatibility_vector is not None:
                best_program["compatibility_vector"] = self.compatibility_vector
                
            if self.agent.diversity_vector is not None:
                best_program = [best_program]
                best_program.extend(self.agent.diversity_vector)
                
            pickle.dump(best_program, f)
            
        


