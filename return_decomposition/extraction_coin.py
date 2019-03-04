import numpy as np

def extraction_coin(seq,rnd_gen,n_features,start_state = 0,fix_state_offset=False,random_start=False,n_batch=1):
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]
    max_timestep = len(seq)
    # Create random actions
    actions = np.asarray(seq, dtype=np.float32)
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

    states_onehot = np.zeros((len(rewards), n_features), dtype=np.float32)
    states_onehot[np.arange(len(rewards)), states] = 1

    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :].tolist(), actions=actions_onehot[None, :].tolist(), rewards=rewards[None, :, None].tolist(),
                true_internal_rewards=true_internal_rewards.tolist())

def extraction_coin_skip(seq,rnd_gen,n_features,start_state = 0,fix_state_offset=False,random_start=False,n_batch=1):
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]
    max_timestep = len(seq)
    # Create random actions
    actions = np.asarray(seq, dtype=np.float32)
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

    states_onehot = np.zeros((len(rewards), n_features), dtype=np.float32)
    states_onehot[np.arange(len(rewards)), states] = 1

    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :].tolist(), actions=actions_onehot[None, :].tolist(), rewards=rewards[None, :, None].tolist(),
                true_internal_rewards=true_internal_rewards.tolist())

def extraction_coin_batch(seqs,rnd_gen,n_features,start_state = 0,fix_state_offset=False,random_start=False,n_batch=2):
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]
    # max_timestep = max(list(map(len,seqs)))
    max_timestep = len(seqs[0])
    actions_onehots = np.zeros((1,max_timestep,2), dtype=np.float32)
    states_onehots = np.zeros((1,max_timestep,n_features), dtype=np.float32)
    rewards = np.zeros((1,max_timestep), dtype=np.float32)
    # seqs = np.tile(seq[np.newaxis,:],(n_batch,1))

    for seq in seqs:
        # Create random actions
        actions = np.asarray(seq, dtype=np.float32)
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
        reward = coin_collect

        # Padd end of game sequences with zero-states
        states = np.asarray(states, np.int) + int(n_features / 2)

        states_onehot = np.zeros((len(reward), n_features), dtype=np.float32)
        states_onehot[np.arange(len(reward)), states] = 1
        actions_onehots = np.concatenate((actions_onehots,actions_onehot[np.newaxis, :]),axis=0)
        states_onehots = np.concatenate((states_onehots,states_onehot[np.newaxis, :]),axis=0)
        rewards = np.concatenate((rewards,reward[np.newaxis, :]),axis=0)
    true_internal_rewards = rewards
    # Return states, actions, and rewards
    return dict(states=states_onehots[1:,:,:], actions=actions_onehots[1:,:,:], rewards=rewards[1:,:,None],
                true_internal_rewards=true_internal_rewards[1:,:,None])

def extraction_coin_batch_skip(seqs,rnd_gen,n_features,start_state = 0,fix_state_offset=False,random_start=False,n_batch=2):
    if random_start:
        # start_state = np.random.randint(-2, 2, 1)[0]
        start_state = np.random.choice([-1, 1], 1, [0.5, 0.5])[0]
    # max_timestep = max(list(map(len,seqs)))
    max_timestep = len(seqs[0])
    actions_onehots = np.zeros((1,max_timestep,2), dtype=np.float32)
    states_onehots = np.zeros((1,max_timestep,n_features), dtype=np.float32)
    rewards = np.zeros((1,max_timestep), dtype=np.float32)
    # seqs = np.tile(seq[np.newaxis,:],(n_batch,1))

    for seq in seqs:
        # Create random actions
        actions = np.asarray(seq, dtype=np.float32)
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
        reward = coin_collect

        # Padd end of game sequences with zero-states
        states = np.asarray(states, np.int) + int(n_features / 2)

        states_onehot = np.zeros((len(reward), n_features), dtype=np.float32)
        states_onehot[np.arange(len(reward)), states] = 1
        actions_onehots = np.concatenate((actions_onehots,actions_onehot[np.newaxis, :]),axis=0)
        states_onehots = np.concatenate((states_onehots,states_onehot[np.newaxis, :]),axis=0)
        rewards = np.concatenate((rewards,reward[np.newaxis, :]),axis=0)
    true_internal_rewards = rewards
    # Return states, actions, and rewards
    return dict(states=states_onehots[1:,:,:], actions=actions_onehots[1:,:,:], rewards=rewards[1:,:,None],
                true_internal_rewards=true_internal_rewards[1:,:,None])