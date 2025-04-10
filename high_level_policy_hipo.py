import os
import time
import pickle
from tqdm import tqdm
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from pretrain.models import ProgramVAE
from rl import utils
from HPRL.pretrain.ppo_iko.model_ppo_option import Flatten, Policy, NNBase
from HPRL.pretrain.ppo_iko.distributions import Categorical
from HPRL.pretrain.ppo_iko.utils import init
from HPRL.pretrain.ppo_iko.storage_option import RolloutStorage
from HPRL.pretrain.ppo_iko.algo import PPO_optionModel

from karel_long_env.karel_long import KarelLong_world
from envs_hipo import make_vec_envs

class PolicyHIPO(Policy):
    def __init__(self, envs, base_kwargs=None):
        nn.Module.__init__(self)
        obs_shape = envs.observation_space.shape
        print(obs_shape)
        if len(obs_shape) == 3:
            base = CNNBaseHIPO
        else:
            raise NotImplementedError()

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, base_kwargs['option_num'])

class CNNBaseHIPO(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=16, input_height=8, input_width=8, option_num=5):
        super(CNNBaseHIPO, self).__init__(recurrent, hidden_size, hidden_size)
        input_shape = (1, num_inputs, input_height, input_width)
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.conv = nn.Sequential(
            nn.utils.weight_norm(init_(nn.Conv2d(num_inputs, 32, 4, stride=1)), name='weight'), nn.ReLU(),
            nn.utils.weight_norm(init_(nn.Conv2d(32, 32, 2, stride=1)), name='weight'), nn.ReLU(), Flatten())

        n_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Sequential(nn.utils.weight_norm(init_(nn.Linear(n_size, hidden_size)), name='weight'), nn.ReLU())
        self.fc2 = nn.Sequential(nn.utils.weight_norm(init_(nn.Linear(option_num, hidden_size)), name='weight'), nn.ReLU())
        self.fc3 = nn.Sequential(nn.utils.weight_norm(init_(nn.Linear(hidden_size*2, hidden_size)), name='weight'), nn.ReLU(),
                                 nn.utils.weight_norm(init_(nn.Linear(hidden_size, hidden_size)), name='weight'), nn.ReLU())
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output_feat = self.conv(input)
        n_size = output_feat.shape[-1]
        return n_size

    def forward(self, inputs, rnn_hxs, masks):
        x = self.conv(inputs[0] / 1.0)
        x = self.fc1(x)
        x2 = self.fc2(inputs[1])
        x = torch.cat([x, x2], -1)
        x = self.fc3(x)

        if self.is_recurrent:
        	x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class RolloutStorageHIPO(RolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, action_shape,
                 recurrent_hidden_state_size, option_num):
        print("RolloutStorage num_steps: {}, num_processes: {}, obs_shape: {}, action_shape: {}".format(num_steps, num_processes, obs_shape, action_shape))
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.one_hot = torch.zeros(num_steps + 1, num_processes, option_num)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.one_hot = self.one_hot.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, one_hot, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.one_hot[self.step + 1].copy_(one_hot)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.one_hot[0].copy_(self.one_hot[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            one_hot_batch = self.one_hot[:-1].view(-1, self.one_hot.size(-1))[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, one_hot_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            one_hot_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                #print(start_ind + offset)
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                one_hot_batch.append(self.one_hot[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            one_hot_batch = torch.stack(one_hot_batch, 1).squeeze(1)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, one_hot_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

class PPOHIPO(PPO_optionModel):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        super(PPOHIPO, self).__init__(actor_critic, clip_param, ppo_epoch, num_mini_batch,
                                      value_loss_coef, entropy_coef, lr, eps,
                                      max_grad_norm, use_clipped_value_loss)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, one_hot_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    (obs_batch, one_hot_batch), recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
 

class ProgramDecoder(nn.Module):
    def __init__(self, envs, config):
        super(ProgramDecoder, self).__init__()
        self.program_vae = ProgramVAE(envs, **config)
        self.program_vae.vae.decoder.setup = "supervised"

    def forward(self, h_meta, deterministic=False):
        z = h_meta
        output = self.program_vae.vae.decoder(None, z, teacher_enforcing=False, deterministic=deterministic,
                                              evaluate=False)

        return output


class HighLevelPolicy(object):
    def __init__(self, device, config, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = dummy_envs
        self.dsl = dsl

        self.algo = config['PPO']['algo']
        self.num_env_steps = config['PPO']['num_env_steps']
        self.num_steps = config['PPO']['num_steps']
        self.num_processes = config['PPO']['num_processes']
        self.use_linear_lr_decay = config['PPO']['use_linear_lr_decay']
        self.decoder_deterministic = config['PPO']['decoder_deterministic']
        self.use_gae = config['PPO']['use_gae']
        self.gae_lambda = config['PPO']['gae_lambda']
        self.gamma = config['PPO']['gamma']
        self.use_proper_time_limits = config['PPO']['use_proper_time_limits']
        
        self.env_name = config['env_name']
        self.log_interval = config['log_interval']
        self.save_interval = config['save_interval']
        self._world = KarelLong_world(env_task=config["env_task"])

        self.save_path = self.config["outdir"]

        custom = True
        logger.info('Using environment: {}, {}'.format(config["env_name"], config["env_task"]))

        self.envs = make_vec_envs(config["env_name"], config['seed'], self.num_processes,
                                  config['gamma'], os.path.join(config['outdir'], 'HighLevelPolicy', 'env'),
                                  device, False, custom_env=custom, custom_kwargs={'config': config['args']})

        self.env_eval = make_vec_envs(config["env_name"], config['seed'], 1,
                                  config['gamma'], os.path.join(config['outdir'], 'HighLevelPolicy', 'env'),
                                  device, False, custom_env=custom, custom_kwargs={'config': config['args']})
        obs = self.envs.reset()

        self.program_decoder = ProgramDecoder(self.envs, config)
        self.program_decoder.to(device)

        checkpt = self.config['net']['saved_params_path']
        if checkpt is not None:
            self.logger.debug('Loading params from {}'.format(checkpt))
            params = torch.load(checkpt, map_location=self.device)
            self.program_decoder.program_vae.load_state_dict(params[0], strict=False)

        if config['CEM']['compatibility_vector'] is not None:
            compatibility_vector = config['CEM']['compatibility_vector']
            with open(compatibility_vector, "rb") as input_file:
                compatibility_vector = pickle.load(input_file)

                if isinstance(compatibility_vector, dict):
                    self.compatibility_vector = [compatibility_vector]

                elif isinstance(compatibility_vector, list):
                    self.compatibility_vector = compatibility_vector
                    
                else:
                    raise NotImplementedError()
                    
            if 'compatibility_vector' in self.compatibility_vector[0].keys():
                self.compatibility_vector = self.compatibility_vector[0:1]
                temp = self.compatibility_vector[0]['compatibility_vector']
                if isinstance(temp, dict):
                    self.compatibility_vector.append(temp)

                elif isinstance(temp, list):
                    self.compatibility_vector.extend(temp)
            else:
            	raise NotImplementedError()
                    
        else:
            raise NotImplementedError()

        option_vector_list = []
        option_program_list = []
        for i in self.compatibility_vector:
            option_vector_list.append(i['vector'])
            option_program_list.append(i['program_str'])
            self.logger.debug('Option {}'.format(i['program_str']))

        option_vector_list.reverse()
        self.option_num = self.config['option_num']
        option_vector_list = option_vector_list[:self.option_num]
        if len(option_vector_list) == 1:
            option_vector_list = option_vector_list * 2
        option_vector_tensor = torch.stack(option_vector_list, dim=0).to(self.device)
        self.pred_programs = self.program_decoder(option_vector_tensor, deterministic=True)[1]

        # Termination option
        self.option_num += 1
        dummy_tensor = torch.tensor([1, 2, 6, 6, 6, 6, 3] + [50]*(config['dsl']['max_program_len'] - 7)).to(self.device).view(1, -1)
        self.pred_programs = torch.cat((self.pred_programs, dummy_tensor), dim=0)

        indices = torch.arange(self.pred_programs.size(0)).unsqueeze(-1).to(self.device)
        self.pred_programs = torch.cat((indices, self.pred_programs), dim=1)[:self.option_num]

        base_kwargs = {
            'recurrent': False,
            'hidden_size': config['PPO']['hidden_size'],
            'input_height': config['height'],
            'input_width': config['width'],
            'option_num': self.option_num
        }

        self.actor_critic = PolicyHIPO(self.envs, base_kwargs=base_kwargs)
        self.actor_critic.to(device)

        if self.config["high_level_policy_checkpt"] is not None:
            high_level_policy_checkpt = self.config["high_level_policy_checkpt"]
            params = torch.load(high_level_policy_checkpt)        
            self.actor_critic.load_state_dict(params, strict=True)

        self.agent = PPOHIPO(
            self.actor_critic,
            config['PPO']['clip_param'],
            config['PPO']['ppo_epoch'],
            config['PPO']['num_mini_batch'],
            config['PPO']['value_loss_coef'],
            config['PPO']['entropy_coef'],
            lr=config['PPO']['lr'],
            eps=config['PPO']['eps'],
            max_grad_norm=config['PPO']['max_grad_norm']
        )

        self.rollouts = RolloutStorageHIPO(
            self.num_steps,
            self.num_processes,
            self.envs.observation_space.shape, 
            1,
            self.actor_critic.recurrent_hidden_state_size,
            self.option_num
        )


        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(device)

    def train(self):
        start = time.time()
        num_updates = int(self.num_env_steps) // self.num_steps // self.num_processes
        pre_reward = torch.zeros(self.num_processes, 1)
        best_valid_reward = 0
        action_count = 0
        val_count = 0

        for j in range(num_updates):
            self.actor_critic.train()
            
            if self.use_linear_lr_decay:    
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.config["PPO"]["lr"])

            for step in range(self.num_steps):
                with torch.no_grad():
                    inputs = (self.rollouts.obs[step], self.rollouts.one_hot[step])
                    value, z, z_log_prob, recurrent_hidden_states = self.actor_critic.act(inputs, self.rollouts.recurrent_hidden_states[step], self.rollouts.masks[step])
                    option_one_hot = F.one_hot(z.squeeze(1), num_classes=self.option_num).float().to(self.device)

                with torch.no_grad():
                    pg_list = []
                    for select in z.flatten().tolist():
                        pg_list.append(self.pred_programs[select])

                pred_programs = torch.stack(pg_list, dim=0).to(self.device)
                obs, reward, done, infos = self.envs.step(pred_programs)
                action_count += sum([info["exec_data"]['a_h_len'] for info in infos])
                done_tensor = torch.from_numpy(done).unsqueeze(1)
                
                with torch.no_grad():
                    add_reward = reward - pre_reward

                    pre_reward = reward * done_tensor.ne(True)
                    option_one_hot[done_tensor.squeeze(1), :] = torch.zeros(self.option_num).to(self.device)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[1.0] for i in range(len(infos))])
                self.rollouts.insert(obs, option_one_hot, recurrent_hidden_states, z, z_log_prob, value, add_reward, masks, bad_masks)

            total_num_steps = (j + 1) * self.num_processes * self.num_steps
            mean_reward = round(reward.mean().cpu().detach().item(), 3)
            min_reward = round(reward.min().cpu().detach().item(), 3)
            max_reward = round(reward.max().cpu().detach().item(), 3)
            self.logger.debug("Last step: mean/min/max reward/total_num_steps/total_action {:.4f}/{:.4f}/{:.4f}, {}, {}".
                         format(mean_reward, min_reward, max_reward, total_num_steps, action_count, end=' '))

            self.writer.add_scalar('train/mean_reward', mean_reward, total_num_steps)
            self.writer.add_scalar('train/max_reward', min_reward, total_num_steps)
            self.writer.add_scalar('train/min_reward', max_reward, total_num_steps)
            self.writer.add_scalar('train/action_count', action_count, total_num_steps)
            

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    (self.rollouts.obs[-1], self.rollouts.one_hot[-1]),
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]
                ).detach()
    
            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda, self.use_proper_time_limits)
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()

            if (j+1) % self.config["PPO"]["val_interval"]  == 0:
                z_count = []
                break_by_option = False
                break_by_limit = False
                self.actor_critic.eval()
                
                self.logger.debug("===========Validation Start===========")
                reward_evals = []
                for k in range(self.config["PPO"]["num_demo_val"]):
                    obs_eval  = self.env_eval.reset()
                    recurrent_hidden_states = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size).to(self.device)
                    masks = torch.ones(1, 1).to(self.device)
                    option_one_hot_eval = torch.zeros(1, self.option_num).to(self.device)
                    for i in tqdm(range(self.config["max_episode_steps"])):
                        inputs = (obs_eval, option_one_hot_eval)
                        with torch.no_grad():
                            value, z_eval, z_log_prob, recurrent_hidden_states = self.actor_critic.act(inputs, recurrent_hidden_states, masks)
                            option_one_hot_eval = F.one_hot(z_eval.squeeze(1), num_classes=self.option_num).float().to(self.device)

                        with torch.no_grad():
                            pg_list = []
                            z_count.extend(z_eval.flatten().tolist())
                            for select in z_eval.flatten().tolist():
                                pg_list.append(self.pred_programs[select])

                            pred_programs = torch.stack(pg_list, dim=0).to(self.device)
                            obs_eval, reward_eval, done_eval, infos_eval = self.env_eval.step(pred_programs)
                            if done_eval[0]:
                                if select == self.option_num-1:
                                    self.logger.debug("Break by Option")
                                    break_by_option = True
                                elif i == self.config["max_episode_steps"] - 1:
                                    self.logger.debug("Break by Limit")
                                    break_by_limit = True	
                                else:
                                    print("Break by Env")
                                break

                            if select == self.option_num-1:
                                    self.logger.debug("Break by Option")
                                    break_by_option = True
                                    break
                                    
                    reward_evals.extend(reward_eval.tolist())
                self.logger.debug("===========Validation End===========")
                
                val_count += 1
                self.logger.debug("Validation mean/best reward/total_num_steps/total_action {:.4f}/{:.4f}, {}, {}:".
                                  format(np.mean(reward_evals).mean(), best_valid_reward, total_num_steps, action_count))
                self.writer.add_scalar('val/mean_reward', np.mean(reward_evals).mean(), total_num_steps)

                if np.mean(reward_evals) >= best_valid_reward and break_by_option:
                    best_valid_reward = np.mean(reward_evals)
                    torch.save(self.actor_critic.state_dict(), os.path.join(self.save_path, self.env_name + "_" + self.config['env_task'] + "_model.pt"))
                    self.logger.debug("Saving model")
