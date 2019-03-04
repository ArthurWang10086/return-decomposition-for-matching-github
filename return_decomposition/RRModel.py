from collections import OrderedDict

import tensorflow as tf
from TeLL.layers import DenseLayer, LSTMLayer, RNNInputLayer, ConcatLayer, ReshapeLayer
from TeLL.utility.misc import make_sure_path_exists
from TeLL.utility.misc_tensorflow import tensor_shape_with_flexible_dim, TriangularValueEncoding
from TeLL.regularization import regularize

class RRModel(object):
    def __init__(self, state_shape, n_actions, max_timestep, n_batch=1,
                 n_lstm_cells=8, regularize_coef=0.,
                 entropy_coef=0., entropy_temperature=0,
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
        # print(tf.shape(rr_returns['predictions']))
        # print(tf.shape(rr_returns['lstm_internals']))
        # print(tf.shape(rr_returns['lstm_h']))
        # print(tf.shape(rr_returns['time']))
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
        return_internal = tf.reshape(rr_returns['predictions'],[-1])
        return_prediction = tf.reduce_sum(rr_returns['predictions'],[1,2])
        true_return = tf.reduce_sum(targets,[1,2])
        AUCW = (true_return*tf.cast(n_timesteps,tf.float32) - tf.reduce_sum(tf.reshape(rr_returns['predictions'][0, :, 0],[-1])*tf.cast((tf.range(n_timesteps+1)[::-1]+1),tf.float32)))/tf.cast(n_timesteps,tf.float32)
        reward_prediction_error = tf.reduce_mean(tf.square(tf.reduce_sum(targets,[1,2])-tf.reduce_sum(rr_returns['predictions'],[1,2])))

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
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
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
            true_ret, pred_ret,return_internal_,AUCW_ = sess.run(
                [true_return, return_prediction,return_internal, AUCW], feed_dict=feed_dict)

            return {'loss': 0, 'true_return': true_ret, 'predict_return': pred_ret,'AUCW':AUCW_,'return_internal':return_internal_}

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