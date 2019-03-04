import argparse
import numpy as np
from collections import OrderedDict
import datetime as dt
import time
import os

import tensorflow as tf
from TeLL.layers import DenseLayer, LSTMLayer, RNNInputLayer, ConcatLayer, ReshapeLayer
from TeLL.utility.misc import make_sure_path_exists
from TeLL.utility.misc_tensorflow import tensor_shape_with_flexible_dim, TriangularValueEncoding
from TeLL.regularization import regularize
from return_decomposition.extraction_coin import extraction_coin,extraction_coin_batch
from return_decomposition.extraction_ball import extraction_ball
# from baselines.ppo2_rudder.reward_redistribution import entropy
import ast
from return_decomposition.RRModel import RRModel
import pandas as pd

# episode 3701: mse 70.9599380493164;rmse 8.423772198327564;avg_loss 0;AUCW 8.673013687133789 avg_gap 4.8313117027282715; ret 99.93937683105469; pred 100.24079895019531;


# 修改内容：12-5
# 1、重构代码，数据集作为文件输入而非内部规则生成
# 2、增加测试集代码
# 3、增加评估指标

# 修改内容：12-9
# 1、完成潮人篮球序列数据生成
# 2、思考state action建模
# 3、思考评价指标
# 4、代码跑起来了

# 修改内容：12-13
# 1、统计了各种行为
# 2、增加了用户画像
# 3、debug为什么投篮加总和最后结果对不上
# 4、解决用户画像人数问题：用排位的比赛
# 5、增加了将胜负改为比分之和的实验
# 6、增加了只用得分行为的实验

# def make_env(scenario_name, arglist):
#     from baselines.ppo2_rudder.multiagent_particle_envs_master.multiagent.environment import MultiAgentEnv
#     import baselines.ppo2_rudder.multiagent_particle_envs_master.multiagent.scenarios as scenarios
#
#     # load scenario from script
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world(arglist)
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
#     return env


def entropy(logits, temperature=1.):
    a0 = tf.cond(tf.reduce_any(tf.less(logits, 0)), lambda: logits + tf.reduce_min(logits, axis=-1, keep_dims=True),
                 lambda: logits)
    a0 = tf.clip_by_value(a0 / temperature, -10, 10)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


class SampleGenerator(object):
    def __init__(self, arglist):
        self.max_episode_len = arglist.max_episode_len
        self.num_advs = arglist.num_adv
        self.episode_step = 0
        self.episode_num = 0

        # Create environment
        self.env = make_env(arglist.scenario, arglist)
        self.obs_shape_each_adv = self.env.observation_space[0].shape[0]
        self.action_num = self.env.action_space[0].n
        print('numer of experimental advs is:{}'.format(self.num_advs))

        self.obs_n = self.env.reset()  # a list, which include the relative info for each agent

    def step(self, action_n):
        # print("episode num:{} step:{}".format(self.episode_num, self.episode_step))

        # environment step
        # print('obs_n:', self.obs_n)
        new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
        # print('rew_n', rew_n)
        # print('new_obs_n:', new_obs_n)

        # re-calculate reward
        rew_n = self.calculate_reward(new_obs_n)

        self.obs_n = new_obs_n.copy()

        self.episode_step += 1
        done = all(done_n)
        terminal = (self.episode_step >= self.max_episode_len)
        if done or terminal:
            # print("Done")
            self.episode_num += 1
            self.reset()

        return new_obs_n, rew_n, done_n, info_n

    def calculate_reward(self, new_obs_n):
        """
        move close +1, move away -1.
        :param new_obs_n:
        :return: new_r: list, reward of each agent.
        """
        new_r = []
        for i in range(self.num_advs):
            distance_square = np.sum(np.square(self.obs_n[i][4:6]))  # [4:6] is relative pos of landmark and agent
            new_distance_square = np.sum(np.square(new_obs_n[i][4:6]))

            if new_distance_square >= distance_square:
                new_r.append(-1)
            else:
                new_r.append(1)

        return new_r

    def reset(self):
        self.episode_step = 0
        self.obs_n = self.env.reset()

    def render(self):
        self.env.render()





def dataset_generate(filename='dataset.txt'):
    import json
    L=[]
    for i in range(int(n_max_updates)):
        L.append(generate_sample3())
    with open(filename,'w') as f:
        f.write('\n'.join(L))

def generate_sample2(padding=False, fix_state_offset=True, random_start=False, *args, **kwargs):
    """Even state move right or odd state move left, give reward 1."""
    global start_state
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]

    # Create random actions
    actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(max_timestep,)), dtype=np.float32)
    actions_onehot = np.zeros((max_timestep, 2), dtype=np.float32)
    actions_onehot[actions == 0, 0] = 1
    actions_onehot[:, 1] = 1 - actions_onehot[:, 0]
    actions += actions - 1
    # Create states to actions, make sure agent stays in range [-6, 6]
    states = np.zeros_like(actions)
    for i, a in enumerate(actions):
        if i == 0:
            # states[i] = a
            states[i] = np.clip(start_state + a, a_min=-int(n_features / 2), a_max=int(n_features / 2))
        else:
            states[i] = np.clip(states[i - 1] + a, a_min=-int(n_features / 2), a_max=int(n_features / 2))

    if fix_state_offset:
        states = np.insert(states, 0, start_state)[:-1]

    # Check when agent collected a coin (=is at position 2)
    # coin_collect = np.asarray(states == 2, dtype=np.float32)
    coin_collect = np.asarray((states % 2 == 0) == (actions == 1), dtype=np.float32)

    true_internal_rewards = coin_collect.copy()
    # Move all reward to position 50 to make it a delayed reward example
    coin_collect[-1] = np.sum(coin_collect)
    coin_collect[:-1] = 0
    rewards = coin_collect

    # Padd end of game sequences with zero-states
    states = np.asarray(states, np.int) + int(n_features / 2)
    if padding:
        states_onehot = np.zeros((len(rewards) + ending_frames, n_features), dtype=np.float32)
        states_onehot[np.arange(len(rewards)), states] = 1
        # actions_onehot = np.concatenate((actions_onehot, np.zeros_like(actions_onehot[:ending_frames])))
        actions_onehot = np.concatenate((actions_onehot, np.zeros(shape=[ending_frames, *actions_onehot.shape[1:]],
                                                                  dtype=actions_onehot.dtype)))
        # rewards = np.concatenate((rewards, np.zeros_like(rewards[:ending_frames])))
        rewards = np.concatenate((rewards, np.zeros(shape=[ending_frames, *rewards.shape[1:]],
                                                    dtype=rewards.dtype)))
        # true_internal_rewards = np.concatenate((true_internal_rewards, np.zeros_like(true_internal_rewards[:ending_frames])))
        true_internal_rewards = np.concatenate((true_internal_rewards,
                                                np.zeros(shape=[ending_frames, *true_internal_rewards.shape[1:]],
                                                         dtype=true_internal_rewards.dtype)))
    else:
        states_onehot = np.zeros((len(rewards), n_features), dtype=np.float32)
        states_onehot[np.arange(len(rewards)), states] = 1

    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :].tolist(), actions=actions_onehot[None, :].tolist(), rewards=rewards[None, :, None].tolist(),
                true_internal_rewards=true_internal_rewards.tolist())

def generate_sample3(padding=False, fix_state_offset=True, random_start=False, *args, **kwargs):
    """Even state move right or odd state move left, give reward 1."""
    global start_state
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]

    # Create random actions
    import random
    actions = rnd_gen.randint(low=0, high=2, size=max_timestep)
    return '0'+'@'+','.join([str(x) for x in actions.tolist()])


def test(filename,args):
    global episode,n_plotted,avg_return,avg_loss,mse

    # Set cuda, -1 disable, 0 enable.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()

    outputpath = os.path.join('workingdir', 'coin_code', "{}_{}".format(
        dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), args.save_name))
    outputpath_training = os.path.join(outputpath, 'training')
    make_sure_path_exists(outputpath)
    make_sure_path_exists(outputpath_training)

    summary_writer = tf.summary.FileWriter(outputpath, graph=tf.get_default_graph())
    print(n_batch)

    model = RRModel(state_shape=n_features, n_actions=2, max_timestep=max_timestep, regularize_coef=args.regularize_coef,
                    entropy_coef=args.entropy_coef, entropy_temperature=args.entropy_temperature,
                    use_time_input=args.use_time_input,n_batch=n_batch)
    f = open(filename,'r')
    return_train  = []
    while episode < n_max_updates:
        seqs = []
        for i in range(n_batch):
            tmp = f.readline()
            if len(tmp)<1:
                f.seek(0)
                tmp = f.readline()
                seqs.append(tmp.split('@')[1].split(','))
            else:
                seqs.append(tmp.split('@')[1].split(','))
        sample = extraction_coin_batch(seqs=np.array(seqs),rnd_gen=rnd_gen,n_features=n_features,n_batch=n_batch)
        # if args.skip_no_reward_trajectory:
        #     if np.all(sample['rewards'] == 0):
        #         continue
        #(1,50,13) (1,50,2) (1,50,1)
        returns = model.train(states=np.array(sample['states']), actions=np.array(sample['actions']), rewards=np.array(sample['rewards']))
        return_train.append(returns)

        import math

        if episode % 100 == 1:
            true_ret = [y for returns in return_train for y in returns['true_return']]
            pred_ret = [y for returns in return_train for y in returns['predict_return']]
            mse_train = np.average([(true_ret[i]-pred_ret[i])*(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            avg_return_train = np.average([abs(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            AUCW_train = np.average([returns['AUCW'] for returns in return_train])
            # print(return_train[0]['true_return'])
            # print(return_train[0]['return_internal'])
            # print(return_train[0]['AUCW'])
            print(
                "episode {}: mse {};rmse {};avg_loss {};AUCW {} avg_gap {}; ret {}; pred {};".format(episode,mse_train,math.sqrt(mse_train),avg_loss,AUCW_train,
                                                                                        avg_return_train,
                                                                                        np.average(true_ret), np.average(pred_ret)))
            return_train  = []

        episode += 1

def test_ball(filename,args):
    global episode,n_plotted,avg_return,avg_loss,mse

    # Set cuda, -1 disable, 0 enable.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()
    from actionEncode import actionEncode
    model = RRModel(state_shape=actionEncode.EncodeMapLen*6, n_actions=actionEncode.EncodeMapLen*6, max_timestep=None, regularize_coef=args.regularize_coef,
                    entropy_coef=args.entropy_coef, entropy_temperature=args.entropy_temperature,
                    use_time_input=args.use_time_input,n_batch=n_batch)
    f = open(filename+'.txt','r')
    f_test = open(filename+'_test.txt','r')
    return_train  = []
    return_test = []
    while episode < n_max_updates:
        tmp = f.readline()
        if len(tmp)<1:
            f.seek(0)
            tmp = f.readline()
        # print(episode,tmp)
        sample = extraction_ball(seq=tmp.split('@')[1].split(','),reward=tmp.split('@')[0])
        returns = model.train(states=sample['states'], actions=sample['actions'], rewards=sample['rewards'])
        return_train.append(returns)
        import math

        if episode % 1000 == 1:
            true_ret = [returns['true_return'] for returns in return_train]
            pred_ret = [returns['predict_return'] for returns in return_train]
            mse_train = np.average([(true_ret[i]-pred_ret[i])*(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            avg_return_train = np.average([abs(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            avg_loss = np.average([returns['loss'] for returns in return_train])
            AUCW_train = np.average([returns['AUCW'] for returns in return_train])
            return_train = []
            # print(tmp)
            print(
                "episode {}: mse {};rmse {};avg_loss {};AUCW {} avg_gap {}; ret {}; pred {};".format(episode,mse_train,math.sqrt(mse_train),avg_loss,AUCW_train,
                                                                                                     avg_return_train,
                                                                                                     true_ret[0],pred_ret[0]))
            # f_test.seek(0)
            # for i in range(30):
            #     tmp = f_test.readline()
            #     sample = extraction_ball(seq=tmp.split('@')[1].split(','),reward=tmp.split('@')[0])
            #     returns = model.test(states=sample['states'], actions=sample['actions'], rewards=sample['rewards'])
            #     return_test.append(returns)
            #
            # true_ret_test = [returns['true_return'] for returns in return_test]
            # pred_ret_test = [returns['predict_return'] for returns in return_test]
            # avg_loss = np.average([returns['loss'] for returns in return_test])
            # mse_test = np.average([(true_ret_test[i]-pred_ret_test[i])*(true_ret_test[i]-pred_ret_test[i]) for i in range(len(true_ret_test))])
            # avg_return_test = np.average([abs(true_ret_test[i]-pred_ret_test[i]) for i in range(len(true_ret_test))])
            # AUCW_test = np.average([returns['AUCW'] for returns in return_test])
            # with open('dataset/ball_internal/2018-11-01.txt','w') as f2:
            #     f2.write('\n'.join([','.join([str(x) for x in returns['return_internal']]) for returns in return_test]))
            # return_test = []
            # print(
            #     "episode {}: mse {};rmse {};avg_loss {};AUCW {} avg_gap {}; ret {}; pred {};".format(episode,mse_test,math.sqrt(mse_test),avg_loss,AUCW_test,
            #                                                                                          avg_return_test,
            #                                                                                          np.average(true_ret_test), np.average(pred_ret_test)))
        episode += 1
    f.close()


def run(args):
    # Set cuda, -1 disable, 0 enable.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()

    model = RRModel(state_shape=10, n_actions=5, max_timestep=3000)
    game = SampleGenerator(args)
    while True:
        game.render()
        time.sleep(0.1)

        action_n = [np.random.randint(0, game.action_num) for _ in range(game.num_advs)]
        # action_n = [3, 0, 0]
        # print('action_n', action_n)
        new_obs_n, rew_n, done_n, info_n = game.step(action_n)

        print("new_r:", rew_n)


if __name__ == '__main__':
    import sys
    # sys.path.append("../")
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument('--level', type=int, choices=[0, 1], default=1,
                        help='choose the level of env, 0: dense reward, 1: sparse reawrd')
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num_adv", type=int, default=3, help="number of adversaries")

    parser.add_argument('--save_name', '--sn', type=str, default='default', help="Save name.")
    parser.add_argument('--entropy_coef', type=float, default=0., help="Entropy coef.")
    parser.add_argument('--entropy_temperature', type=float, default=1, help="Entropy coef.")
    parser.add_argument('--regularize_coef', type=float, default=1e-6, help="Entropy coef.")

    parser.add_argument('--skip_no_reward_trajectory', type=ast.literal_eval, default=False,
                        help="skip_no_reward_trajectory")
    parser.add_argument('--use_time_input', type=ast.literal_eval, default=False, help="use_time_input")
    # parser.add_argument('--batch', type=ast.literal_eval, default=1, help="batch")
    parser.add_argument("--batch", type=int, default=1, help="number of batch")
    parser.add_argument("--max_timestep", type=int, default=10, help="number of max_timestep")
    parser.add_argument("--n_features", type=int, default=13, help="number of n_features")
    parser.add_argument("--n_max_updates", type=int, default=50000, help="number of n_max_updates")

    args, unknown = parser.parse_known_args()
    print('addition_params', args)

    rnd_seed = 123
    rnd_gen = np.random.RandomState(seed=rnd_seed)
    tf.set_random_seed(rnd_seed)
    max_timestep = args.max_timestep
    n_mb = 1
    n_features = args.n_features
    n_actions = 2
    ending_frames = 10
    avg_return = 0.
    n_max_updates = args.n_max_updates
    episode = 0
    n_plotted = 0
    last_continue_plot_num = 20
    avg_loss = 0
    start_state = 0
    mse = 0
    n_batch = args.batch
    # dataset_generate('dataset/coin/dataset.txt')
    # test('dataset/coin/dataset.txt',args)
    test_ball('dataset/ball/2018-11-01.diff',args)