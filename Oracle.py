import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor

'''
Collection of different contextual multi-armed bandits

Nice explanation of MAB: https://stats.stackexchange.com/questions/291906/can-reinforcement-learning-be-stateless

Author: Zubow (TU Berlin)
'''

class OracleHeuristicBandit:
    '''
    Some kind of oracle heuristic ...
    '''
    def __init__(self, agent_id, num_bs, num_antennas):
        self.agent_type = 1
        self.agent_id = agent_id
        self.num_bs = num_bs
        self.num_antennas = num_antennas
        self.max_nulls = min(num_antennas - 1, num_bs - 1)
        self.min_angular_separation = 5.0 / 360 # 5 degree

    def predict(self, context):
        # own beam direction
        beam_dir = context[self.agent_id,0]
        # sort the STAs from OBSS according to their distance; closest first
        ap_context = context[self.agent_id,2:]
        # 1-col = angle; 2-col=pathloss
        sta_ctx = np.reshape(ap_context, (self.num_bs - 1, 2))
        # sort desc
        sorted_sta_ctx = sta_ctx[sta_ctx[:, 1].argsort()[::-1]]
        # STAs to null
        nulls = []
        for nu in range(sorted_sta_ctx.shape[0]):
            # check if bf angle and nulling are not too close
            if np.abs(sorted_sta_ctx[nu, 0] - beam_dir) > self.min_angular_separation:
                nulls.append(sorted_sta_ctx[nu, 0])
            else:
                nulls.append(np.nan)
            if len(nulls) == self.num_antennas - 1:
                break

        return nulls

    def update(self, action, context, reward):
        # do nothing
        pass