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
from extraction_coin import extraction_coin

# from baselines.ppo2_rudder.reward_redistribution import entropy
import ast
import pandas as pd


# 修改内容：
# 1、重构代码，数据集作为文件输入而非内部规则生成
# 2、增加测试集代码
# 3、增加评估指标

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


class RRModel(object):
    def __init__(self, state_shape, n_actions, max_timestep, n_batch=1,
                 n_lstm_cells=8, regularize_coef=0.,
                 entropy_coef=0., entropy_temperature=1,
                 use_time_input=False):
        """
        Reward redistribute model.
        :param state_shape: the shape of state, e.g. (dim1, dim2...)
        :param n_batch:
        :param use_time_input:
        """
        print("use_time_input:", use_time_input)
        sess = tf.get_default_session()
        # Set up reward redistribution model


        # This will encode the in-game time in multiple nodes
        timestep_encoder = None
        if use_time_input:
            timestep_encoder = TriangularValueEncoding(max_value=max_timestep, triangle_span=int(max_timestep / 10))

        states_placeholder = tf.placeholder(shape=(n_batch, max_timestep, state_shape), dtype=tf.float32)
        actions_placeholder = tf.placeholder(shape=(n_batch, max_timestep, n_actions), dtype=tf.float32)
        rewards_placeholder = tf.placeholder(shape=(n_batch, max_timestep, 1), dtype=tf.float32)
        state_shape_per_ts = (n_batch, 1, state_shape)
        action_shape_per_ts = (n_batch, 1, n_actions)
        # true_internal_rewards_placeholder = tf.placeholder(shape=(n_batch, None, 1), dtype=tf.float32)
        n_timesteps = tf.shape(rewards_placeholder)[1] - 1

        with tf.variable_scope('reward_redistribution_model', reuse=tf.AUTO_REUSE):
            state_input_layer = RNNInputLayer(tf.zeros(state_shape_per_ts, dtype=tf.float32))
            action_input_layer = RNNInputLayer(tf.zeros(action_shape_per_ts, dtype=tf.float32))
            if use_time_input:
                time_input_layer = RNNInputLayer(timestep_encoder.encode_value(tf.constant(0, dtype=tf.int32)))
                time_input = ReshapeLayer(incoming=time_input_layer, shape=(n_batch, 1, timestep_encoder.n_nodes_python))
                reward_redistibution_input = ConcatLayer(incomings=[state_input_layer, action_input_layer, time_input],
                                                         name='RewardRedistributionInput')
            else:
                reward_redistibution_input = ConcatLayer(incomings=[state_input_layer, action_input_layer],
                                                         name='RewardRedistributionInput')
            # n_lstm_cells = 8
            truncated_normal_init = lambda mean, stddev: \
                lambda *args, **kwargs: tf.truncated_normal(mean=mean, stddev=stddev, *args, **kwargs)
            w_init = truncated_normal_init(mean=0, stddev=1)
            og_bias = truncated_normal_init(mean=0, stddev=1)
            ig_bias = truncated_normal_init(mean=-1, stddev=1)
            ci_bias = truncated_normal_init(mean=0, stddev=1)
            fg_bias = truncated_normal_init(mean=-5, stddev=1)
            lstm_layer = LSTMLayer(incoming=reward_redistibution_input, n_units=n_lstm_cells,
                                   name='LSTMRewardRedistribution',
                                   W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                                   b_ci=ci_bias([n_lstm_cells]), b_ig=ig_bias([n_lstm_cells]),
                                   b_og=og_bias([n_lstm_cells]), b_fg=fg_bias([n_lstm_cells]),
                                   a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.identity,
                                   c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, store_states=True)

            n_output_units = 1
            output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, a=tf.identity, W=w_init,
                                      b=tf.zeros([n_output_units], dtype=tf.float32), name="OutputLayer")

        lstm_input_shape = reward_redistibution_input.get_output_shape()

        # Ending condition
        def cond(time, *args):
            """Break if game is over by looking at n_timesteps"""
            return ~tf.greater(time, n_timesteps)

        # Loop body
        # Create initial tensors
        init_tensors = OrderedDict([
            ('time', tf.constant(0, dtype=tf.int32)),
            ('lstm_inputs', tf.zeros(lstm_input_shape)),
            ('lstm_internals', tf.expand_dims(tf.stack([lstm_layer.c[-1], lstm_layer.c[-1],
                                                        lstm_layer.c[-1], lstm_layer.c[-1],
                                                        lstm_layer.c[-1]], axis=-1), axis=1)),
            ('lstm_h', tf.expand_dims(lstm_layer.h[-1], axis=1)),
            ('predictions', tf.zeros([s if s >= 0 else 1 for s in output_layer.get_output_shape()]))
        ])

        # Get initial tensor shapes in tf format
        init_shapes = OrderedDict([
            ('time', init_tensors['time'].get_shape()),
            ('lstm_inputs', tensor_shape_with_flexible_dim(init_tensors['lstm_inputs'], dim=1)),
            ('lstm_internals', tensor_shape_with_flexible_dim(init_tensors['lstm_internals'], dim=1)),
            ('lstm_h', tensor_shape_with_flexible_dim(init_tensors['lstm_h'], dim=1)),
            ('predictions', tensor_shape_with_flexible_dim(init_tensors['predictions'], dim=1)),
        ])

        def body(time, lstm_inputs, lstm_internals, lstm_h, predictions, *args):
            """Loop over states and additional inputs, compute network outputs and store hidden states and activations for
            debugging/plotting"""

            # Set states as network input
            state_input_layer.update(tf.expand_dims(states_placeholder[:, time], axis=1))

            # Set actions as network input
            action_input_layer.update(tf.cast(tf.expand_dims(actions_placeholder[:, time], axis=1), dtype=tf.float32))

            # Set time as network input
            if use_time_input:
                time_input_layer.update(timestep_encoder.encode_value(time))

            # Update LSTM cell-state and output with states from last timestep
            lstm_layer.c[-1], lstm_layer.h[-1] = lstm_internals[:, -1, :, -1], lstm_h[:, -1, :]

            # Calculate reward redistribution network output and append it to last timestep
            predictions = tf.concat([predictions, output_layer.get_output()], axis=1)

            # Store LSTM states for all timesteps for visualization
            # TODO: add option on this to save memory.
            lstm_internals = tf.concat([lstm_internals,
                                        tf.expand_dims(tf.stack([lstm_layer.ig[-1], lstm_layer.og[-1],
                                                                 lstm_layer.ci[-1], lstm_layer.fg[-1],
                                                                 lstm_layer.c[-1]], axis=-1), axis=1)],
                                       axis=1)
            lstm_h = tf.concat([lstm_h, tf.expand_dims(lstm_layer.h[-1], axis=1)], axis=1)

            # Increment time
            time += tf.constant(1, dtype=tf.int32)

            lstm_inputs = tf.concat([lstm_inputs, reward_redistibution_input.out], axis=1)

            return [time, lstm_inputs, lstm_internals, lstm_h, predictions]

        wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=tuple(init_tensors.values()),
                               shape_invariants=tuple(init_shapes.values()),
                               parallel_iterations=50, back_prop=True, swap_memory=True)

        # Re-Associate returned tensors with keys
        rr_returns = OrderedDict(zip(init_tensors.keys(), wl_ret))

        # Remove initialization timestep
        rr_returns['lstm_internals'] = rr_returns['lstm_internals'][:, 1:]
        rr_returns['lstm_h'] = rr_returns['lstm_h'][:, 1:]
        rr_returns['predictions'] = rr_returns['predictions'][:, 1:]
        lstm_inputs = rr_returns['lstm_inputs'][:, 1:]

        #
        # Define updates
        #
        # aux_target_1_filter = tf.ones((10, 1, 1), dtype=tf.float32)
        # aux_target_1 = tf.concat([rewards_placeholder, tf.zeros_like(rewards_placeholder[:, :9])], axis=1)
        # aux_target_1 = tf.nn.conv1d(aux_target_1, filters=aux_target_1_filter, padding='VALID', stride=1)
        # aux_target_2 = tf.reduce_sum(rewards_placeholder, axis=1) - tf.cumsum(rewards_placeholder, axis=1)
        # aux_target_3 = tf.cumsum(rewards_placeholder, axis=1)
        # targets = tf.concat([rewards_placeholder, aux_target_1, aux_target_2, aux_target_3], axis=-1)
        targets = rewards_placeholder

        return_prediction = tf.reduce_sum(rr_returns['predictions'][0, :, 0])
        true_return = tf.reduce_sum(targets[0, :, 0])
        AUCW = true_return*max_timestep - tf.reduce_sum(tf.reshape(rr_returns['predictions'][0, :, 0],[-1])*tf.cast((tf.range(max_timestep)[::-1]+1),tf.float32))
        reward_prediction_error = tf.square(true_return - return_prediction)
        # auxiliary_losses = tf.reduce_mean(tf.square(targets[0, :, 1:] - rr_returns['predictions'][0, :, 1:]),
        #                                   axis=1)

        total_loss = reward_prediction_error

        # Add regularization penalty
        if regularize_coef > 0:
            rr_reg_penalty = regularize(layers=[lstm_layer, output_layer], l1=regularize_coef,
                                        regularize_weights=True, regularize_biases=True)
            total_loss = (reward_prediction_error) + rr_reg_penalty

        # Add entropy
        if entropy_coef > 0:
            predicted_reward_entropy = entropy(logits=rr_returns['predictions'][0, :, 0],
                                               temperature=entropy_temperature)
            total_loss = total_loss - predicted_reward_entropy * entropy_coef

        trainables = tf.trainable_variables()
        grads = tf.gradients(total_loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        rr_update = optimizer.apply_gradients(zip(grads, trainables))

        # TF summary
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("reward_prediction_error", reward_prediction_error)
        # tf.summary.scalar("internal_reward_predict_error", tf.reduce_sum(
        #     tf.square(true_internal_rewards_placeholder - rr_returns['predictions'][0, :, 0])))
        summaries = tf.summary.merge_all()

        if regularize_coef > 0:
            tf.summary.scalar("rr_reg_penalty", rr_reg_penalty)

        if entropy_coef > 0:
            tf.summary.scalar("predicted_reward_entropy", predicted_reward_entropy)

        def train(states, actions, rewards, ):
            feed_dict = {states_placeholder: states,
                         actions_placeholder: actions,
                         rewards_placeholder: rewards}
            loss, true_ret, pred_ret, _,AUCW_, summary = sess.run(
                [total_loss, true_return, return_prediction, rr_update,AUCW, summaries], feed_dict=feed_dict)

            return {'loss': loss, 'true_return': true_ret, 'predict_return': pred_ret,
                    'summaries': summary,'AUCW':AUCW_}

        def test(states, actions, rewards, ):
            feed_dict = {states_placeholder: states,
                         actions_placeholder: actions,
                         rewards_placeholder: rewards}
            true_ret, pred_ret,AUCW_ = sess.run(
                [true_return, return_prediction, AUCW], feed_dict=feed_dict)

            return {'loss': 0, 'true_return': true_ret, 'predict_return': pred_ret,'AUCW':AUCW_}

        def step(states, actions, rewards):
            feed_dict = {states_placeholder: states,
                         actions_placeholder: actions,
                         rewards_placeholder: rewards}
            prediction_values = sess.run(rr_returns['predictions'], feed_dict=feed_dict)

            return prediction_values

        self.train = train
        self.test = test
        self.step = step
        tf.global_variables_initializer().run(session=sess)


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
    actions = rnd_gen.randint(low=0, high=2, size=(max_timestep,))
    return '0'+'@'+','.join([str(x) for x in actions.tolist()])


def test(args):
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

    model = RRModel(state_shape=13, n_actions=2, max_timestep=50, regularize_coef=args.regularize_coef,
                    entropy_coef=args.entropy_coef, entropy_temperature=args.entropy_temperature,
                    use_time_input=args.use_time_input)
    f = open('dataset2.txt','r')
    f_test = open('dataset_test.txt','r')
    import  json
    # test_data = [json.loads(x) for x in f_test.read().split('\n')]
    return_train  = []
    return_test = []
    while episode < n_max_updates:
        # sample = generate_sample2()
        #batch_size = 1
        # sample = json.loads(f.readline())
        sample = extraction_coin(seq=f.readline().split('@')[1].split(','),rnd_gen=rnd_gen,n_features=n_features)
        if args.skip_no_reward_trajectory:
            if np.all(sample['rewards'] == 0):
                continue
        #(1,50,13) (1,50,2) (1,50,1)
        returns = model.train(states=np.array(sample['states']), actions=np.array(sample['actions']), rewards=np.array(sample['rewards']))
        return_train.append(returns)


        # summary_writer.add_summary(summary, episode)
        import math

        if episode % 100 == 1:
            true_ret = [returns['true_return'] for returns in return_train]
            pred_ret = [returns['predict_return'] for returns in return_train]
            mse_train = np.average([(true_ret[i]-pred_ret[i])*(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            avg_return_train = np.average([abs(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            AUCW_train = np.average([returns['AUCW'] for returns in return_train])

            # loss, true_ret, pred_ret, summary, AUCW_ = returns['loss'], returns['true_return'], returns['predict_return'], returns['summaries'],returns['AUCW']
            # avg_return += float(abs(true_ret-pred_ret))
            # avg_loss = avg_loss * 0.99 + loss * 0.01
            # mse += float(true_ret-pred_ret)*(true_ret-pred_ret)
            print(
                "episode {}: mse {};rmse {};avg_loss {};AUCW {} avg_gap {}; ret {}; pred {};".format(episode,mse_train,math.sqrt(mse_train),avg_loss,AUCW_train,
                                                                                        avg_return_train,
                                                                                        np.average(true_ret), np.average(pred_ret)))
            return_train  = []
            return_test = []
            # for i in range(10000):
            #     sample = json.loads(f_test.readline())
            #     returns = model.test(states=np.array(sample['states']), actions=np.array(sample['actions']), rewards=np.array(sample['rewards']))
            #     return_test.append(returns)
            # true_ret = [returns['true_return'] for returns in return_test]
            # pred_ret = [returns['predict_return'] for returns in return_test]
            # mse_test = np.average([(true_ret[i]-pred_ret[i])*(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            # avg_return_test = np.average([abs(true_ret[i]-pred_ret[i]) for i in range(len(true_ret))])
            # print(
            #     "Test episode {}: mse {};rmse {};avg_loss {};AUCW {}, avg_gap {}; ret {}; pred {};".format(0,mse_test,math.sqrt(mse_test),avg_loss,AUCW_,
            #                                                                                   avg_return_test,
            #                                                                                   np.average(true_ret), np.average(pred_ret)))

        if episode > 5000 and episode % 1000 == 0 or (
                episode + last_continue_plot_num >= n_max_updates):
            print("\tperforming IG, plotting curves...")

            n_plotted += 1

            prediction_values = model.step(states=sample['states'], actions=sample['actions'],
                                           rewards=sample['rewards'])
            states = np.argmax(sample['states'][0], axis=-1) - int(n_features / 2)
            actions = np.argmax(sample['actions'][0], axis=-1)

            pd_data = pd.DataFrame(
                {'state': states, 'action': actions, 'reward': np.squeeze(sample['rewards'][0], axis=-1),
                 'true_internal_rewards': sample['true_internal_rewards'],
                 'predict_internal_rewards': prediction_values[0, ..., 0]})
            pd_data.to_csv(os.path.join(outputpath, 'save_datas_ep{}.csv'.format(episode)), index=False)
        episode += 1


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

    model = RRModel(state_shape=10, n_actions=5, max_timestep=3)
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
    parser.add_argument('--use_time_input', type=ast.literal_eval, default=True, help="use_time_input")

    args, unknown = parser.parse_known_args()
    print('addition_params', args)

    rnd_seed = 123
    rnd_gen = np.random.RandomState(seed=rnd_seed)
    tf.set_random_seed(rnd_seed)
    max_timestep = 50
    n_mb = 1
    n_features = 13
    n_actions = 2
    ending_frames = 10
    avg_return = 0.
    n_max_updates = 5000
    episode = 0
    n_plotted = 0
    last_continue_plot_num = 20
    avg_loss = 0
    start_state = 0
    mse = 0
    dataset_generate('../dataset/coin/dataset2.txt')
    # run(args)
    # test(args)