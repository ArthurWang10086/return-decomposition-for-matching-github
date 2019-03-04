import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandasql import  sqldf
import actionEncode

def plot_bar(df):
    plt.figure()
    # df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
    df.plot.bar()
    plt.show()

if __name__ == '__main__':
    reward = open('../dataset/ball_internal/2018-11-01.txt','r').read().split('\n')
    data = open('../dataset/ball/2018-11-01.diff_test.txt','r').read().split('\n')
    assert len(data[0].split('@')[1].split(',')) == len(reward[0].split(','))

    # action_list = [str(y.split(':')[0] in ['0','1','2'])+'#'.join(y.split(':')[1].split('#')[:3]) for x in [data[i].split('@')[1].split(',') for i in range(len(reward))] for y in x]
    # print(len(action_list))
    # reward_list = [y for x in [reward[i].split(',') for i in range(len(reward))] for y in x]
    # df = pd.DataFrame({"actions":action_list,"reward":reward_list})
    # print(sqldf("SELECT actions,AVG(reward) FROM df GROUP BY actions;", locals()))
    i=1
    reward_list = [float(y) for y in reward[i].split(',')]
    action_list = [y.split(':')[0] in ['0','1','2'] for y in data[i].split('@')[1].split(',')]
    print(reward_list)
    print(action_list)
    df = pd.DataFrame({"lose_reward":[reward_list[i] if action_list[i] else 0 for i in range(len(action_list))],"win_reward":[reward_list[i] if not action_list[i] else 0 for i in range(len(action_list))]})
    plot_bar(df)

