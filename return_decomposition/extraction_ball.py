import numpy as np
from actionEncode import actionEncode
def extraction_ball(seq,reward=0):
    max_timestep = len(seq)
    actions_onehots = np.zeros((max_timestep,1), dtype=np.float32)
    states_onehots = np.zeros((max_timestep,1), dtype=np.float32)
    rewards = np.zeros((max_timestep, 1), dtype=np.float32)
    for encodefunc in actionEncode.EncodeMapList:
        players = [int(x.split(':')[0]) for x in seq]
        actions = np.asarray([encodefunc(x.split(':')[1]) for x in seq], dtype=np.float32)
        actions_onehot = np.zeros((max_timestep, encodefunc.size*6), dtype=np.float32)
        states_onehot = np.zeros((max_timestep, encodefunc.size*6), dtype=np.float32)
        for i in range(max_timestep):
            actions_onehot[i][int(actions[i])+players[i]*encodefunc.size] = 1
            states_onehot[i] = states_onehot[i-1]
            states_onehot[i][int(actions[i])+players[i]*encodefunc.size] += 1
        actions_onehots = np.concatenate((actions_onehots,actions_onehot),axis=1)
        states_onehots = np.concatenate((states_onehots,states_onehot),axis=1)
    rewards[max_timestep-1] = reward
    true_internal_rewards = rewards
    # Return states, actions, and rewards
    return dict(states=states_onehots[None, :,1:], actions=actions_onehots[None,:,1:], rewards=rewards[None, :],
                true_internal_rewards=true_internal_rewards)

if __name__ == '__main__':
    datas = open('../dataset/ball/2018-11-01.txt').read().split('\n')
    for data in datas:
        result  = data.split('@')[0]
        seq = data.split('@')[1].split(',')
        try:
            actions = np.asarray(seq, dtype=np.float32)
        except Exception:
            print(data)
