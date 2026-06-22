import csv
import numpy as np
import random

import helper
from BS import BS
from STA import STA, path_loss_lin
from ULA import ULA
import gym
from gym import spaces
import matplotlib.pyplot as plt

"""
    Simulation Framework
    observation space: (|BS| x 2*|BS|) -> for each BS: angle, normalized pathloss towards each STA (0: own, rest: in OBSS)
    action space:      (|BS| x MIN(N_ANT-1,|BS|-1)) -> set of nulling angles for each BS
"""
class Sim(gym.Env):
    def __init__(self, config):
        """
        :param config: the full configuration of the simulation
        """
        super(Sim, self).__init__()
        self.debug = False
        self.num_bs = config.num_bss
        self.min_num_stas = config.min_num_stas
        self.max_num_stas = config.max_num_stas
        self.client_radius = config.client_radius
        self.bw_mhz = config.bw_mhz
        self.channel_freq = config.channel_freq
        self.num_antennas = config.num_antennas
        self.tick = config.tick
        self.channel_update_interval_in_ticks = config.channel_update_interval_in_ticks
        self.move_prob = config.move_prob
        # max steps in episode: 1k corresponds to 5s of simulation time ... STAs can move by up to 7m
        self.max_steps_episode = config.max_steps_episode

        self.rx_noise_fig = 6 # noise figure at receiver
        self.max_eirp = 20  # tx power in DL at BS in dBm
        self.max_num_stas_p_bs = 0  # updated after assignment of stas to bs
        self.num_steps = 0
        self.min_dist_to_bs = 1.5  # m
        self.DataRatev1 = []
        self.DataRatev2 = []
        self.Oracle_DataRatev1 = []
        self.Oracle_DataRatev2 = []
        self.flag = True
        self.flag2 = True
        self.Oracle_flag = True
        self.state = np.zeros((self.num_bs, self.num_antennas - 1)) - 1
        self.number_of_sta_per_bs = []
        self.place_nodes()

        # everything normalized to 1 (i.e. angle/2pi, pathloss)
        low_obs = np.array([[0.0, 0.0] * self.num_bs] * self.num_bs)
        high_obs = np.array([[1.0, 1.0] * self.num_bs] * self.num_bs)
        low_act = np.array([[0.0] * min(self.num_antennas - 1, self.num_bs-1)] * self.num_bs)
        high_act = np.array([[1.0] * min(self.num_antennas - 1, self.num_bs-1)] * self.num_bs)

        self.observation_space = spaces.Box(low_obs, high_obs) # angle of arrival, normalized pathloss
        self.action_space = spaces.Box(low_act, high_act) # nulling angles
        self.prev_reward = 0.0
        self.Oracle_prev_reward = 0.0

        if self.debug:
            print('mimoSim init ...')

    def get_spaces(self):
        n_features = self.observation_space.shape[0] * self.observation_space.shape[1]
        #print("Size of Feature Space ->  %d x %d" % (self.observation_space.shape[0], self.observation_space.shape[1]))
        n_actions = self.action_space.shape[0] * self.action_space.shape[1]
        #print("Size of continuous Action Space ->  {}".format(n_actions))
        return (n_actions, n_features)

    def place_nodes(self):
        self.set_bss_position() # place BSs

        self.bss: list[BS] = []
        self.stas: list[STA] = []
        for n, pos in enumerate(self.bs_pos):
            num_stas = random.randrange(self.min_num_stas, self.max_num_stas + 1)
            if num_stas > self.max_num_stas_p_bs:
                self.max_num_stas_p_bs = num_stas
            bs = BS(n, pos, self.rx_noise_fig, num_stas, self.client_radius, self.bw_mhz,
                    self.num_antennas, self.channel_freq, self.max_eirp, len(self.stas), self.tick, self.move_prob,
                    self.min_dist_to_bs)
            self.stas += bs.stas
            self.number_of_sta_per_bs.append(len(bs.stas))
            self.bss.append(bs)
        for bs in self.bss:
            bs.update_sta_pos(0, self.stas)

    def get_observation(self):
        obs = np.empty((self.num_bs, self.num_bs * 2))
        next_sta_ids = []
        for bs in self.bss:
            next_sta_ids.append(bs.stas[self.num_steps % bs.num_stas].idx)
        for bs in self.bss:
            tmp = bs.relative_sta_pos[next_sta_ids, :]
            wanted_dir = tmp[bs.idx,:]
            # normalize angle
            wanted_dir[0] = wanted_dir[0] / 360.0
            # convert dist to normalized pathloss
            wanted_dir[1] = self.dist_to_normalized_pl(wanted_dir[1])
            inf_dir = tmp[~np.isin(np.arange(len(tmp)), bs.idx),:]

            # number of interferers = |BS|-1
            assert inf_dir.shape[0] == self.num_bs - 1
            for jj in range(inf_dir.shape[0]):
                inf_dir[jj,0] = inf_dir[jj,0] / 360.0 # normalize angle
                inf_dir[jj,1] = self.dist_to_normalized_pl(inf_dir[jj,1])
            assert np.logical_and(inf_dir[:,1] >= 0, inf_dir[:,1] <= 1).all()

            combined = np.concatenate((wanted_dir, inf_dir.flatten(order='C')), axis=0)
            # obs: wanted_angle, wanted_norm_pl, inf_angle_1, inf_norm_pl_1, inf_angle_2, inf_norm_pl_2, ...
            obs[bs.idx,:] = combined

        assert obs.shape[0] == self.num_bs and obs.shape[1] == self.num_bs * 2
        return obs

    def dist_to_normalized_pl(self, dist):
        '''
        Normalized to 0-1 with PL higher than 100dB set to zero
        :param dist:
        :return:
        '''
        pl_db = helper.to_dB(path_loss_lin(dist, self.channel_freq + (self.bw_mhz * 1e6 / 2)))
        thr = -100
        normalized_pl = max(pl_db, thr)
        normalized_pl = (pl_db - thr) / (abs(thr) / 2)
        return normalized_pl

    def step(self, action: np.ndarray, Oracle_action: np.ndarray):
        """
        :param action: |BS| x n_antenna - 1 in [0, 360]
        :return:
        """
        curr_time = self.num_steps * self.tick

        # check format for Oracle
        assert Oracle_action.shape[0] == self.num_bs
        assert Oracle_action.shape[1] == min(self.num_bs-1, self.num_antennas-1)
        assert np.isnan(np.max(Oracle_action)) or np.max(Oracle_action) <= 1.0
        assert np.isnan(np.max(Oracle_action)) or np.min(Oracle_action) >= 0.0
        active_stas = []
        for i, null in enumerate(Oracle_action):
            active_stas.append(self.bss[i].stas[self.num_steps % self.bss[i].num_stas])
            beam_ang = self.bss[i].relative_sta_pos[active_stas[-1].idx, 0]
            #print('BS: %d bf: %.2f, null: %.2f' % (i, beam_ang, null))
            assert beam_ang >= 0 and beam_ang <= 360.0
            if null.shape[0] == 1 and np.isnan(null[0]): # np.isnan(np.sum(null)):
                self.bss[i].calc_weights_pure_bf(beam_ang)
            else:
                # remove nan
                null = null[~np.isnan(null)]
                # convert into degrees
                # if null < 0:
                #     null = null * 360.0 * (-1)
                # else:
                null = null * 360.0
                assert np.min(null) >= 0 and np.max(null) <= 360.0
                self.bss[i].calc_weights(beam_ang, null)

        if self.debug:
            print("Active STAs ->  {}".format(active_stas))

        rates = []
        for sta in active_stas:
            sta.calc_snir(self.bss, False)
            Rate = sta.calc_rates(self.tick, False)
            rates.append(Rate)
        
        alpha = 0.1
        Oracle_reward = 0.
        reward_raw = 0.
        if np.size(np.where(np.isclose(rates, 0))) == 0:
            # reward = np.sum(np.log2(rates)) # sum of log() to account for fairness
            reward_raw = np.sum(np.log2(rates))
            interference = max(0, self.Oracle_prev_reward - reward_raw)
            Oracle_reward = reward_raw - alpha * interference
            self.Oracle_prev_reward = reward_raw

        self.Oracle_DataRatev1.append(Oracle_reward)
        # if (self.num_steps + 1) % 100 == 0:
        #     with open('Oracle_Reward_'+'_BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)+ '_'+ str(self.max_num_stas) +'_' + str(0) +'.csv', 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         if self.Oracle_flag == True:
        #             writer.writerow(['Oracle_Reward_'+'BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)])
        #             self.Oracle_flag = False
        #         writer.writerow([round((sum(self.Oracle_DataRatev1)/len(self.Oracle_DataRatev1)),2)])
                
        #############################





        # check format for DDPG
        assert action.shape[0] == self.num_bs
        assert action.shape[1] == min(self.num_bs-1, self.num_antennas-1)
        assert np.isnan(np.max(action)) or np.max(action) <= 1.0
        assert np.isnan(np.max(action)) or np.min(action) >= 0.0

        

        active_stas = []
        for i, null in enumerate(action):
            active_stas.append(self.bss[i].stas[self.num_steps % self.bss[i].num_stas])
            beam_ang = self.bss[i].relative_sta_pos[active_stas[-1].idx, 0]
            #print('BS: %d bf: %.2f, null: %.2f' % (i, beam_ang, null))
            assert beam_ang >= 0 and beam_ang <= 360.0
            if null.shape[0] == 1 and np.isnan(null[0]): # np.isnan(np.sum(null)):
                self.bss[i].calc_weights_pure_bf(beam_ang)
            else:
                # remove nan
                null = null[~np.isnan(null)]
                # convert into degrees
                null = null * 360.0
                assert np.min(null) >= 0 and np.max(null) <= 360.0
                self.bss[i].calc_weights(beam_ang, null)

        if self.debug:
            print("Active STAs ->  {}".format(active_stas))

        rates = []
        for sta in active_stas:
            sta.calc_snir(self.bss, False)
            Rate = sta.calc_rates(self.tick, False)
            rates.append(Rate)
        
        rates = np.array(rates)

        alpha = 0.1
        reward = 0.
        reward_raw = 0.
        if np.size(np.where(np.isclose(rates, 0))) == 0:
            # reward = np.sum(np.log2(rates)) # sum of log() to account for fairness
            reward_raw = np.sum(np.log2(rates))
            interference = max(0, self.prev_reward - reward_raw)
            reward = reward_raw - alpha * interference
            self.prev_reward = reward_raw
        
        self.DataRatev1.append(reward)
        if (self.num_steps + 1) % 100 == 0:
            print('Mean of last %d-> DDPG: %.2f \t Oracle: %.2f' % (len(self.DataRatev1), sum(self.DataRatev1)/len(self.DataRatev1), sum(self.Oracle_DataRatev1)/len(self.Oracle_DataRatev1)))
            self.DataRatev2.append(sum(self.DataRatev1)/len(self.DataRatev1))
            self.Oracle_DataRatev2.append(sum(self.Oracle_DataRatev1)/len(self.Oracle_DataRatev1))
            self.DataRatev1 = []
            self.Oracle_DataRatev1 = []


        if (len(self.DataRatev2) + 1) % 50 == 0:
            with open('Average_DDPG_Reward_'+'_BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)+ '_'+ str(self.max_num_stas) +'_' + str(0) +'.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                if self.flag2 == True:
                    writer.writerow(['DDPG_Reward_'+'BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)])
                    self.flag2 = False
                writer.writerow([round((sum(self.DataRatev2)/len(self.DataRatev2)),2)])
                
            with open('Average_Oracle_Reward_'+'_BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)+ '_'+ str(self.max_num_stas) +'_' + str(0) +'.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                if self.Oracle_flag == True:
                    writer.writerow(['Oracle_Reward_'+'BS_' + str(self.num_bs) + '_Ant_' + str(self.num_antennas)])
                    self.Oracle_flag = False
                writer.writerow([round((sum(self.Oracle_DataRatev2)/len(self.Oracle_DataRatev2)),2)])
            
            self.DataRatev2 = []
            self.Oracle_DataRatev2 = []

        for s in self.stas:
            s.update_position(curr_time, self.client_radius, self.min_dist_to_bs)

        self.num_steps += 1

        if self.num_steps % self.channel_update_interval_in_ticks == 0:
            if self.debug:
                print("... update CSI")

            for bs in self.bss:
                bs.update_sta_pos(curr_time, self.stas)

        obs = self.get_observation()

        if self.debug:
            print('%.4f, Sim::step %d, reward: %.2f' % (curr_time, self.num_steps, reward))

        done = False
        if self.num_steps >= 2000:
            done = True

        return obs, reward, done, {}
    
    def render3(self):
        """
        :param null: per BS in deg
        :return:
        """

        x = list()
        y = list()
        ax = plt.gca()
        bsX = []
        bsY = []
        color = []
        
        for bs in self.bss:
            x.append(bs.pos[0])
            y.append(bs.pos[1])
            bsX.append(bs.pos[0])
            bsY.append(bs.pos[1])
            color.append(next(ax._get_lines.prop_cycler)['color'])
        plt.scatter(x, y, c=color, marker='p', s=100)
        x = list()
        y = list()

        stX = []
        stY = []
        for sta in self.stas: 
            x.append(sta.pos[0])
            y.append(sta.pos[1])
            stX.append(sta.pos[0])
            stY.append(sta.pos[1])
        
        
        BS_list = []
        i = 0
        for bs in self.bss:
            BS_list.append('bs' + str(i))
            i = i + 1
        BS_dictionary = {}
        i = 0
        for bs in BS_list:
            BS_dictionary[bs] = [bsX[i],bsY[i]]
            i = i + 1


        STA_list = []
        i = 0
        for sta in self.stas:
            STA_list.append('sta' + str(i))
            i = i + 1
        STA_dictionary = {}
        i = 0
        for bs in STA_list:
            STA_dictionary[bs] = [stX[i],stY[i]]
            i = i + 1
        
        staColor = []
        zz = 0
        for i in range(len(bsX)):
            z = 0
            c = color[i]
            for j in range(len(stX)):
                if z < self.number_of_sta_per_bs[i]:
                    staColor.append(c)
                    x1, y1 = [], []
                    x1.append(bsX[i])
                    x1.append(stX[zz])
                    y1.append(bsY[i])
                    y1.append(stY[zz])
                    # dist = round(math.dist([bsY[i],stX[zz]], [ bsX[i], stY[zz]]),1)
                    # plt.text(stX[zz],stY[zz],str(dist))
                    # plt.plot(x1, y1, c=c)
                    z = z + 1
                    zz = zz + 1
                    
        
        plt.scatter(x, y, c=staColor)
        plt.xlim(-30,30)
        plt.ylim(-30,30)
        plt.pause(0.05)
        plt.cla()

    def render(self):
        """
        :param null: per BS in deg
        :return:
        """
        fig, ax = plt.subplots()

        # plot BSs
        x = list()
        y = list()
        colors = np.arange(0, len(self.bss), 1, dtype=int)
        for bs in self.bss:
            x.append(bs.pos[0])
            y.append(bs.pos[1])

        self.sc = ax.scatter(x, y, c=colors, marker='p', s=100)

        # plot STAs
        x = list()
        y = list()
        colors = list()
        for sta in self.stas:
            x.append(sta.pos[0])
            y.append(sta.pos[1])
            print('STA %d: %.2f,%.2f' % (sta.idx, sta.pos[0], sta.pos[1]))
            colors.append(sta.bs.idx)

        self.sc = ax.scatter(x, y, c=colors)

        plt.xlim((-25, 25))
        plt.ylim((-25, 25))
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()

    def reset(self):
        
        # new placement for nodes
        self.number_of_sta_per_bs = []
        self.place_nodes()
        # reset steps
        self.num_steps = 0
        curr_time = self.num_steps * self.tick
        for s in self.stas:
            s.update_position(curr_time, self.client_radius, self.min_dist_to_bs)
        for bs in self.bss:
            bs.update_sta_pos(curr_time, self.stas)

        return self.get_observation()

    def set_bss_position(self):
        if self.num_bs == 2:
            self.bs_pos = np.array([[0., 0., 0.],
                            [16., 0., 0.]
                            ])
        elif self.num_bs == 3:
            self.bs_pos = np.array([[0., 0., 0.],
                            [np.cos(1 / 3 * np.pi) * 16., np.sin(1 / 3 * np.pi) * 16., 0.],
                            [16., 0., 0.]
                            ])
        elif self.num_bs == 7:
            self.bs_pos = np.array([[0., 0., 0.],
                            [16., 0., 0.],
                            [np.cos(1 / 3 * np.pi) * 16., np.sin(1 / 3 * np.pi) * 16., 0.],
                            [np.cos(2 / 3 * np.pi) * 16., np.sin(2 / 3 * np.pi) * 16., 0.],
                            [-16., 0., 0.],
                            [np.cos(4 / 3 * np.pi) * 16., np.sin(4 / 3 * np.pi) * 16., 0.],
                            [np.cos(5 / 3 * np.pi) * 16., np.sin(5 / 3 * np.pi) * 16., 0.]
                            ])
        else:
            assert False, "Only 2, 3, and 7 BS supported for now"
            

def test_ula():
    freq = 5150e6
    ula_dist = helper.c / (2 * freq)
    num_antennas = 4
    ula = ULA(num_antennas, ula_dist, np.array((1000, 20, 500)), 0, freq)
    beam_dir = np.array([50])
    null_dir = np.array([120])
    ula.plot_pattern(beam_dir, null_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_ula()

