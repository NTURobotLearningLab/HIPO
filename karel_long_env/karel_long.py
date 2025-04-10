import numpy as np
import copy
from leaps.karel_env.karel import Karel_world

MAX_NUM_MARKER = 1
SCALE = 1
COST = 0.0001 / SCALE

state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
}

action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}

class KarelLong_world(Karel_world):
    def __init__(self, s=None, env_task="infinite_harvester"):
        if s is not None:
            self.set_new_state(s)
            
        self.env_task = env_task

    def set_new_state(self, s, metadata=None):
        self.elapse_step = 0
        self.perception_count = 0
        self.progress_ratio = 0.0
        self.s = s.astype(np.bool_)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]
        self.pos_h = [tuple(self.get_location()[:2])]
        self.r_h = []
        self.d_h = []
        self.progress_h = []
        self.program_reward = 0.0
        self.done = False
        self.metadata = copy.deepcopy(metadata)
        self.total_markers = np.sum(s[:,:,6:])

    def clear_history(self):
        self.perception_count = 0
        self.elapse_step = 0
        self.progress_ratio = 0.0
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.p_v_h = []
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)

        self.r_h = []
        self.progress_h = []
        self.d_h = []
        self.program_reward = 0.0
        self.prev_pos_reward = 0.0
        self.init_pos_reward = 0.0
        self.done = False
        self.total_markers = np.sum(self.s_h[-1][:,:,6:])

    def add_to_history(self, a_idx, agent_pos):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())
        self.elapse_step += 1
        reward, done = self._get_state_reward(agent_pos)
        pos_tuple = tuple(agent_pos[:2])
        self.pos_h.append(pos_tuple)               
        self.done = self.done or done
        self.r_h.append(reward)
        self.progress_h.append(self.progress_ratio)
        self.d_h.append(done)
        self.program_reward += reward
        self.total_markers = np.sum(self.s[:,:,6:])

    def _get_infinite_harvester_task_reward(self, agent_pos):
        done = False
        current_progress_ratio = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        marker_picked = self.metadata["marker_picked"]
        current_progress_ratio = marker_picked / self.metadata["max_marker"]
        reward = current_progress_ratio - self.progress_ratio - COST
        self.progress_ratio = current_progress_ratio
        done = (current_progress_ratio >= 1)

        reward = reward if self.env_task == 'infinite_harvester' else float(done)
        self.done = self.done or done
        
        return reward, done

    def _get_seesaw_task_reward(self, agent_pos):
        if self.done:
            return 0.0 - COST, self.done

        current_progress_ratio = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]
        room_l = self.metadata["room_length"]
        corridor_l = self.metadata["corridor_length"]

        marker_picked = self.metadata["marker_picked"]
        current_progress_ratio = marker_picked / self.metadata["max_marker"]
        reward = current_progress_ratio - self.progress_ratio - COST
        self.progress_ratio = current_progress_ratio
        done = (current_progress_ratio >= 1) or len(self.s_h) >= self.metadata["max_episode_length"]
        
        # update left and right room
        marker_pos = None
        if agent_pos[1] < (1+room_l): # in the left room
            if reward > 0: # pickMarker
                right_row  = self.metadata['right_room_row']
                right_col  = self.metadata['right_room_col']
                marker_pos = (np.random.randint(right_row[0], right_row[1]), np.random.randint(right_col[0], right_col[1]))
                self.s[marker_pos[0], marker_pos[1], 6] = True
                self.s[marker_pos[0], marker_pos[1], 5] = False
            
        elif agent_pos[1] >= (1+room_l+corridor_l): # in the right room
            if reward > 0: # pickMarker
                # random sample position in the left room
                left_row  = self.metadata['left_room_row']
                left_col  = self.metadata['left_room_col']
                marker_pos = (np.random.randint(left_row[0], left_row[1]), np.random.randint(left_col[0], left_col[1])) 
                self.s[marker_pos[0], marker_pos[1], 6] = True
                self.s[marker_pos[0], marker_pos[1], 5] = False

        reward = reward if self.env_task == 'seesaw' else float(done)
        self.done = self.done or done
        
        # unmark the marker after done
        if (self.done or done) and marker_pos:
            self.s[marker_pos[0], marker_pos[1], 6] = False
            self.s[marker_pos[0], marker_pos[1], 5] = True

        return reward, done

    def _get_upNdown_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0 - COST, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        current_pos  = (agent_pos[0], agent_pos[1])
        valid_pos    = self.metadata['agent_valid_positions']
        position_penalty = self.metadata['position_penalty']
        goal_pos = self.metadata['up_marker_pos']
        if self.metadata['reached_marker'] % 2 == 1:
            goal_pos = self.metadata['down_marker_pos']
        
        assert goal_pos in valid_pos 
        # check if reach goal
        new_goal_pos = None
        if current_pos == goal_pos:
            self.metadata['reached_marker'] += 1
            if goal_pos == self.metadata['up_marker_pos']:
                # add penalty for upper valid position with 
                for i in range(len(valid_pos)):
                    if valid_pos[i][0] < current_pos[0]:
                        self.metadata['position_penalty'][valid_pos[i]] = 0
                    else:
                        self.metadata['position_penalty'][valid_pos[i]] = 0.0
                candidate_pos = []
                for pos in valid_pos:
                    if pos[0] > goal_pos[0]:
                        candidate_pos.append(pos)
                assert len(candidate_pos) > 0
                new_goal_pos = candidate_pos[np.random.randint(0, len(candidate_pos))]
                self.metadata['down_marker_pos'] = new_goal_pos
            else:
                # add penalty for lower valid position with 
                for i in range(len(valid_pos)):
                    if valid_pos[i][0] > current_pos[0]:
                        self.metadata['position_penalty'][valid_pos[i]] = 0
                    else:
                        self.metadata['position_penalty'][valid_pos[i]] = 0.0
                candidate_pos = []
                for pos in valid_pos:
                    if pos[0] < goal_pos[0]:
                        candidate_pos.append(pos)
                assert len(candidate_pos) > 0
                new_goal_pos = candidate_pos[np.random.randint(0, len(candidate_pos))]
                self.metadata['up_marker_pos'] = new_goal_pos
        if new_goal_pos:
            self.s[goal_pos[0], goal_pos[1], 6] = False
            self.s[goal_pos[0], goal_pos[1], 5] = True
            self.s[new_goal_pos[0], new_goal_pos[1], 6] = True
            self.s[new_goal_pos[0], new_goal_pos[1], 5] = False
            assert new_goal_pos[0] != goal_pos[0] 
            assert new_goal_pos[0] != current_pos[0] 

        # calculate reward
        current_progress_ratio = self.metadata['reached_marker'] / self.metadata["max_marker_reach"]
        reward = current_progress_ratio - self.progress_ratio - COST
        reward += position_penalty[current_pos]
        self.progress_ratio = current_progress_ratio
        done = (current_progress_ratio >= 1) or len(self.s_h) >= self.metadata["max_episode_length"]
        reward = reward if self.env_task == 'upNdown' else float(done)
        self.done = self.done or done
        
        # unmark the marker after done
        if (self.done or done) and new_goal_pos:
            self.s[new_goal_pos[0], new_goal_pos[1], 6] = False
            self.s[new_goal_pos[0], new_goal_pos[1], 5] = True

        return reward, done

    def _get_farmer_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0 - COST, self.done

        current_progress_ratio = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]
        
        marker_picked = self.metadata["marker_picked"]
        marker_placed = self.metadata["marker_put"]
        current_progress_ratio = (marker_picked + marker_placed) / (self.metadata["max_marker"])
        reward = current_progress_ratio - self.progress_ratio - COST
        self.progress_ratio = current_progress_ratio
        done = False

        # create miracle marker
        total_markers = np.sum(self.s[:,:,6:])
        if total_markers == 0:
            self.metadata['clean_up_iter'] += 1
            if self.metadata['clean_up_iter'] < self.metadata['max_clean_up_iter']:
                self.s[1, w-2, 6] = True
                self.metadata['harvester_mode']    = False
                self.metadata['continuous_marker'] = 1
            else:
                done = True
 
        reward = reward if self.env_task == 'farmer' else float(done)
        self.done = self.done or done
        
        return reward, done

    def _get_infinite_doorkey_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0 - COST, self.done

        current_progress_ratio = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]
        current_pos = (agent_pos[0], agent_pos[1])
        current_action = self.a_h[-1]
        next_target = None

        if self.metadata['current_stage'] == 0 and current_pos == self.metadata['key_1'] and current_action == 3:
            self.metadata['marker_picked'] += 1
            door_pos = self.metadata['door_1']
            self.s[door_pos[0], door_pos[1], 4] = False
            close_pos = self.metadata['door_4']
            self.s[close_pos[0], close_pos[1], 4] = True
            next_target = self.metadata['target_1']
            self.s[next_target[0], next_target[1], 6] = True
            self.metadata['current_stage'] = (self.metadata['current_stage']+1) % 4
        elif self.metadata['current_stage']== 1 and current_pos == self.metadata['target_1'] and current_action == 4:
            self.metadata['marker_picked'] += 1
            door_pos = self.metadata['door_2']
            self.s[door_pos[0], door_pos[1], 4] = False
            self.s[current_pos[0], current_pos[1], 6] = False
            close_pos = self.metadata['door_1']
            self.s[close_pos[0], close_pos[1], 4] = True
            next_target = self.metadata['key_2']
            self.s[next_target[0], next_target[1], 6] = True
            self.metadata['current_stage'] = (self.metadata['current_stage']+1) % 4
        elif self.metadata['current_stage']== 2 and current_pos == self.metadata['key_2'] and current_action == 3:
            self.metadata['marker_picked'] += 1
            door_pos = self.metadata['door_3']
            self.s[door_pos[0], door_pos[1], 4] = False
            close_pos = self.metadata['door_2']
            self.s[close_pos[0], close_pos[1], 4] = True
            next_target = self.metadata['target_2']
            self.s[next_target[0], next_target[1], 6] = True
            self.metadata['current_stage'] = (self.metadata['current_stage']+1) % 4
        elif self.metadata['current_stage']== 3 and current_pos == self.metadata['target_2'] and current_action == 4:
            self.metadata['marker_picked'] += 1
            door_pos = self.metadata['door_4']
            self.s[door_pos[0], door_pos[1], 4] = False
            self.s[current_pos[0], current_pos[1], 6] = False
            close_pos = self.metadata['door_3']
            self.s[close_pos[0], close_pos[1], 4] = True
            next_target = self.metadata['key_1']
            self.s[next_target[0], next_target[1], 6] = True
            self.metadata['current_stage'] = (self.metadata['current_stage']+1) % 4
 
        marker_picked = self.metadata["marker_picked"]
        current_progress_ratio = marker_picked / (self.metadata["max_marker"])
        reward = current_progress_ratio - self.progress_ratio - COST
        self.progress_ratio = current_progress_ratio
        done = (current_progress_ratio >= 0.99)
        
        # unmark the marker after done
        if (self.done or done) and next_target:
            self.s[next_target[0], next_target[1], 6] = False

        reward = reward if self.env_task == 'infinite_doorkey' else float(done)
        self.done = self.done or done
 
        return reward, done

    def _get_state_reward(self, agent_pos):
        if self.env_task == 'infinite_harvester':
            reward, done = self._get_infinite_harvester_task_reward(agent_pos)
        elif self.env_task == 'seesaw':
            reward, done = self._get_seesaw_task_reward(agent_pos)
        elif self.env_task == 'upNdown':
            reward, done = self._get_upNdown_task_reward(agent_pos)
        elif self.env_task == 'farmer':
            reward, done = self._get_farmer_task_reward(agent_pos)
        elif self.env_task == 'infinite_doorkey':
            reward, done = self._get_infinite_doorkey_task_reward(agent_pos)
        else:
            raise NotImplementedError('{} task not yet supported'.format(self.env_task))

        return reward, done

    def state_transition(self, a):
        a_idx = np.argmax(a)
        loc = self.get_location()

        if a_idx == 0:
            # move
            if self.front_is_clear():
                front_loc = self.get_neighbor('front')
                loc_vec = self.s[loc[0], loc[1], :4]
                self.s[front_loc[0], front_loc[1], :4] = loc_vec
                self.s[loc[0], loc[1], :4] = np.zeros(4) > 0
                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0

                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0
                next_loc = front_loc
            else:
                loc_vec = np.zeros(4) > 0
                loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
                self.s[loc[0], loc[1], :4] = loc_vec
                next_loc = loc
            self.add_to_history(a_idx, next_loc)
            
        elif a_idx == 1 or a_idx == 2:
            # turn left or right
            loc_vec = np.zeros(4) > 0
            loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
            self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx, loc)

        elif a_idx == 3 or a_idx == 4:
            # pick up or put a marker
            num_marker = np.argmax(self.s[loc[0], loc[1], 5:])
            
            # just clip the num of markers for now
            if self.env_task in ['seesaw', 'infinite_doorkey']:
                if a_idx == 4:
                    new_num_marker = num_marker
                else:
                    new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER)
            
            elif self.env_task == 'farmer':
                if self.done:
                    new_num_marker = num_marker
                elif a_idx == 4 and self.metadata['harvester_mode']:
                    if self.metadata['continuous_marker'] >= 1.0 * self.metadata['continuous_marker_max']:
                        new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER)
                        self.metadata['continuous_marker'] = 1
                        self.metadata['harvester_mode'] = False
                    else:
                        new_num_marker = num_marker
                elif a_idx == 3 and not self.metadata['harvester_mode']:
                    if self.metadata['continuous_marker'] >= self.metadata['threshold'] * self.metadata['continuous_marker_max']:
                        new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER)
                        self.metadata['continuous_marker'] = 0
                        self.metadata['harvester_mode'] = True
                    else:
                        new_num_marker = num_marker
                else:
                    new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER)
                    
            else:
                new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER)

            picked_num = num_marker - new_num_marker 
            placed_num = new_num_marker - num_marker
            marker_vec = np.zeros(MAX_NUM_MARKER+1+1) > 0
            marker_vec[new_num_marker] = True
            self.s[loc[0], loc[1], 5:] = marker_vec

            if self.env_task == 'infinite_harvester':
                if picked_num > 0 and self.metadata["marker_placed"][loc[0], loc[1]] > 0:
                    self.metadata["marker_placed"][loc[0], loc[1]] -= 1
                    self.metadata["marker_picked"] += 1
                    
                    r, c = np.where(self.s[:, :, 5])
                    valid_marker_pos = np.array(list(zip(r, c)))
                    
                    if len(valid_marker_pos) == 0:
                        pass
                        
                    else:
                        marker_poses = valid_marker_pos[np.random.choice(len(valid_marker_pos), size=picked_num, replace=False)]

                        for marker_pos in marker_poses:
                            if np.random.uniform() < self.metadata["marker_prob"]:
                                self.metadata["marker_placed"][marker_pos[0], marker_pos[1]] += 1

                                if self.s[marker_pos[0], marker_pos[1], 5]:
                                    self.s[marker_pos[0], marker_pos[1], 5] = False
                                    self.s[marker_pos[0], marker_pos[1], 6] = True

                                else:
                                    raise
                                    
            if self.env_task == 'seesaw':
                self.metadata['marker_picked'] += picked_num

            if self.env_task in ['farmer']:
                if not self.metadata['harvester_mode'] and a_idx == 4 and placed_num > 0:
                    self.metadata['seed_number'] -= 1
                    self.metadata['marker_put'] += 1
                    self.metadata['continuous_marker'] += 1
                elif self.metadata['harvester_mode'] and a_idx == 3 and picked_num > 0:
                    self.metadata['marker_picked'] += 1
                    self.metadata['continuous_marker'] += 1
                    self.metadata['seed_number'] += 1
            
            # this must be done after all 'if'
            self.add_to_history(a_idx, loc)

        else:
            raise RuntimeError("Invalid action")
        return