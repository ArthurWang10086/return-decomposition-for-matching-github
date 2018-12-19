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
from extraction_coin import extraction_coin,extraction_coin_batch
from extraction_ball import extraction_ball
# from baselines.ppo2_rudder.reward_redistribution import entropy
import ast
from RRModel import RRModel
import pandas as pd
def test(args):
    rnd_seed = 123
    rnd_gen = np.random.RandomState(seed=rnd_seed)
    tf.set_random_seed(rnd_seed)

    # Set cuda, -1 disable, 0 enable.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()

    max_timestep = 1000
    n_mb = 1
    n_features = 13
    n_actions = 2
    ending_frames = 10

    outputpath = os.path.join('workingdir', 'coin_code', "{}_{}".format(
        dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), args.save_name))
    outputpath_training = os.path.join(outputpath, 'training')
    make_sure_path_exists(outputpath)
    make_sure_path_exists(outputpath_training)
    avg_return = 0.
    n_max_updates = 1e5
    episode = 0
    n_plotted = 0
    last_continue_plot_num = 20
    avg_loss = 0
    start_state = 0
    summary_writer = tf.summary.FileWriter(outputpath, graph=tf.get_default_graph())

    def generate_sample2(padding=False, fix_state_offset=True, random_start=False, *args, **kwargs):
        """Even state move right or odd state move left, give reward 1."""
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
        return dict(states=states_onehot[None, :], actions=actions_onehot[None, :], rewards=rewards[None, :, None],
                    true_internal_rewards=true_internal_rewards)

    model = RRModel(state_shape=13, n_actions=2, max_timestep=max_timestep, regularize_coef=args.regularize_coef,
                    entropy_coef=args.entropy_coef, entropy_temperature=args.entropy_temperature,
                    use_time_input=args.use_time_input)

    return_train  = []
    while episode < n_max_updates:
        sample = generate_sample2()
        if args.skip_no_reward_trajectory:
            if np.all(sample['rewards'] == 0):
                continue
        #(1,50,13) (1,50,2) (1,50,1)
        returns = model.train(states=sample['states'], actions=sample['actions'], rewards=sample['rewards'])
        return_train.append(returns)

        import math

        if episode % 20 == 1:
            true_ret = [returns['true_return'] for returns in return_train]
            pred_ret = [returns['predict_return'] for returns in return_train]
            mse_train = np.average([(true_ret[i]-pred_ret[i])*(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            avg_return_train = np.average([abs(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            AUCW_train = np.average([returns['AUCW'] for returns in return_train])

            print(
                "episode {}: mse {};rmse {};avg_loss {};AUCW {} avg_gap {}; ret {}; pred {};".format(episode,mse_train,math.sqrt(mse_train),avg_loss,AUCW_train,
                                                                                                     avg_return_train,
                                                                                                     np.average(true_ret), np.average(pred_ret)))
            return_train  = []

        episode += 1

if __name__ == '__main__':
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

    args, unknown = parser.parse_known_args()
    print('addition_params', args)

    rnd_seed = 123
    rnd_gen = np.random.RandomState(seed=rnd_seed)
    tf.set_random_seed(rnd_seed)
    max_timestep = 30
    n_mb = 1
    n_features = 201
    n_actions = 2
    ending_frames = 10
    avg_return = 0.
    n_max_updates = 5000000
    episode = 0
    n_plotted = 0
    last_continue_plot_num = 20
    avg_loss = 0
    start_state = 0
    mse = 0
    n_batch = 1
    # dataset_generate('../dataset/coin/dataset.txt')
    # test('../dataset/coin/dataset2.txt',args)
    # test_ball('../dataset/ball/2018-11-01.diff',args)
    test(args)