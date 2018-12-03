import numpy as np

def extraction_ball(seq,rnd_gen,n_features,reward=0,action_dim=2,start_state = 0,fix_state_offset=False,random_start=False,batchsize=1):
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]
    max_timestep = len(seq)
    actions = np.asarray(seq, dtype=np.float32)
    actions_onehot = np.zeros((max_timestep, action_dim), dtype=np.float32)
    states_onehot = np.zeros((max_timestep, action_dim), dtype=np.float32)
    rewards = np.zeros((max_timestep, 1), dtype=np.float32)
    for i in range(max_timestep):
        actions_onehot[i][actions[i]] = 1
        states_onehot[i] = states_onehot[i-1]
        states_onehot[i][actions[i]] += 1
    rewards[max_timestep-1] = reward
    true_internal_rewards = rewards
    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :].tolist(), actions=actions_onehot[None, :].tolist(), rewards=rewards[None, :, None].tolist(),
                true_internal_rewards=true_internal_rewards.tolist())