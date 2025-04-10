import numpy as np
import karel_long
from leaps.karel_env.generator import KarelStateGenerator

class KarelLongStateGenerator(KarelStateGenerator):
    def __init__(self, seed=42):
        super().__init__(seed)

    def generate_single_state_infinite_harvester(self, h=16, w=16):
        s = np.zeros([h, w, len(karel_long.state_table)]) > 0

        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        s[1:h-1, 1:w-1, 6] = True

        marker_prob = 1 - (1/(2*karel_long.SCALE))
        max_marker = int((h-2)*(w-2) / (1 - marker_prob))
        marker_placed = np.zeros([h, w])
        marker_placed[1:h-1, 1:w-1] = 1

        metadata = {"marker_prob": marker_prob, "marker_placed": marker_placed, "max_marker": max_marker, "marker_picked":0}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_seesaw(self, h=16, w=16):
        s = np.zeros([h, w, len(karel_long.state_table)]) > 0
        
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        agent_pos = ((h//2), 1)
        s[agent_pos[0], agent_pos[1], 1] = True
        room_length     = (w-2) // 3
        room_height     = 4
        corridor_length = (w-2) - 2*room_length
        corridor_height = 2 #(h-2) // 3

        # left room wall
        s[1:h-1, 1:1+room_length, 4] = True
        # left room
        s[(h//2)-(room_height//2):(h//2)+(room_height//2), 1:1+room_length, 4] = False

        # corridor wall
        s[1:h-1, 1+room_length:1+room_length+corridor_length, 4] = True
        # corridor
        s[(h//2)-(corridor_height//2):(h//2)+(corridor_height//2), 1+room_length:1+room_length+corridor_length, 4] = False

        # right room wall
        right_start_col = 1 + room_length + corridor_length
        s[1:h-1, right_start_col:right_start_col+room_length, 4] = True
        
        # right room
        s[(h//2)-(room_height//2):(h//2)+(room_height//2), right_start_col:right_start_col+room_length, 4] = False

        # marker
        s[(h//2)-2, 3, 6] = True
        
        seesaw_round = 64
        max_marker = seesaw_round*karel_long.SCALE
        max_episode_length = (corridor_length + room_height*room_length*4) * max_marker * seesaw_round
        
        metadata = {
                "max_marker": max_marker, 
                "max_episode_length": max_episode_length * 100 * karel_long.SCALE, 
                "marker_picked":0, 
                "room_length": room_length,
                "room_height": room_height,
                "corridor_length": corridor_length,
                "corridor_height": corridor_height,
                "left_room_row": [(h//2)-(room_height//2),(h//2)+(room_height//2)],
                "left_room_col": [1,1+room_length],
                "right_room_row":[(h//2)-(room_height//2),(h//2)+(room_height//2)],
                "right_room_col":[right_start_col,right_start_col+room_length],
                }
        
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_upNdown(self, h=8, w=8, constant_penalty=0.0005 / karel_long.SCALE):
        s = np.zeros([h, w, len(karel_long.state_table)]) > 0
        
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        
        # Valid position and penalty position
        position_penalty = {}
        for c in range(1, w-1):
            position_penalty[(h-1-c,c)] = 0.0
        for c in range(1, w-2):
            position_penalty[(h-2-c,c)] = 0.0
        for r in range(3, h-1):
            for c in range(1, w-r):
                position_penalty[(h-r-c), c] = -1.0 * constant_penalty 

        c = 2
        r = h - 1
        valid_agent_pos = []
        valid_init_pos = []
        while r > 0 and c < w:
            s[r, c, 4] = True
            s[r - 1, c, 4] = True
            if r - 1 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 1, c - 1))
                valid_init_pos.append((r - 1, c - 1))
                assert not s[r - 1, c - 1, 4] , "there shouldn't be a wall at {}, {}".format(r - 1, c - 1)
            if r - 2 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 2, c - 1))
                assert not s[r - 2, c - 1, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c - 1)
            if r - 2 > 0 and c > 0:
                valid_agent_pos.append((r - 2, c))
                valid_init_pos.append((r - 2, c))
                assert not s[r - 2, c, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c)
            c += 1
            r -= 1

        agent_valid_positions = list(set(valid_agent_pos))
        valid_init_pos = sorted(list(set(valid_init_pos)), key=lambda x: x[1])

        agent_pos = (h-3, 2)
        s[agent_pos[0], agent_pos[1], 1] = True

        up_marker_pos = (1, w-2)
        down_marker_pos = (h-2, 1)
        
        marker_pos = up_marker_pos
        s[:, :, 5] = True
        s[marker_pos[0], marker_pos[1], 5] = False
        s[marker_pos[0], marker_pos[1], 6] = True

        assert np.sum(s[:, :, 6]) == 1
        
        und_round = 100
        metadata = {
                'agent_valid_positions': agent_valid_positions, 
                'up_marker_pos': up_marker_pos,
                'down_marker_pos': down_marker_pos,
                'max_marker_reach': und_round * karel_long.SCALE,
                'max_episode_length': und_round * (w+h) * 2 * und_round * karel_long.SCALE * 100, 
                'reached_marker': 0,
                'position_penalty': position_penalty,
                'constant_penalty': -1.0 * constant_penalty ,
                }
        
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_farmer(self, h=8, w=8, max_clean_up_iter=10*karel_long.SCALE):
        s = np.zeros([h, w, len(karel_long.state_table)]) > 0

        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        s[1, w-2, 6] = True

        marker_prob = 0.5
        max_marker = ((h-2)*(w-2)) * 2 * max_clean_up_iter

        metadata = {
                "marker_prob": marker_prob, 
                "max_marker": max_marker, 
                "marker_picked":0, 
                "marker_put": 0,
                "seed_number": 0, 
                "harvester_mode": False,
                "threshold": 1 - 1 / max(h-2, w-2),
                "continuous_marker": 1,
                "continuous_marker_max": (h-2)*(w-2),
                "clean_up_iter": 0,
                "max_clean_up_iter": max_clean_up_iter,
                }

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    def generate_single_state_infinite_doorkey(self, h=8, w=8):
        s = np.zeros([h, w, len(karel_long.state_table)]) > 0
        
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        
        s[(h//2), :, 4] = True
        s[:, (w//2), 4] = True


        init_pos = (self.rng.randint(1, (h//2)-1),   self.rng.randint(1, (w//2)-1))
        key_1    = (self.rng.randint(1, (h//2)),     self.rng.randint(1, (w//2)))
        target_1 = (self.rng.randint(1, (h//2)),     self.rng.randint((w//2)+1, w-1))
        key_2    = (self.rng.randint((h//2)+1, h-1), self.rng.randint((w//2)+1, w-1))
        target_2 = (self.rng.randint((h//2)+1, h-1), self.rng.randint(1, (w//2)))
        
        s[key_1[0], key_1[1], 6] = True
        s[target_1[0], target_1[1], 6] = True
        s[key_2[0], key_2[1], 6] = True
        s[target_2[0], target_2[1], 6] = True
        
        door_1 = (2, (w//2))
        door_2 = ((h//2), w-2)
        door_3 = (h-2, (w//2))
        door_4 = ((h//2), 2)

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        assert np.sum(s[:, :, 6]) == 4
        id_round = 100
        metadata = {
                'door_1': door_1,
                'door_2': door_2,
                'door_3': door_3,
                'door_4': door_4,
                'key_1': key_1, 
                'target_1': target_1,
                'key_2': key_2, 
                'target_2': target_2,
                'max_marker': id_round*karel_long.SCALE,
                'marker_picked': 0,
                'current_stage': 0,
                }

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

