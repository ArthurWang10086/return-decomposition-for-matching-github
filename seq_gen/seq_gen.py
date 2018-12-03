#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

# Block   blocked_player  skill_id  result
# Rebound   result  skill_id  rebound_type
# Steal  result stealed_player skill_id
# Pass   result receiver_id skill_id  pass_type
# Skill  skill_id
# Shoot  skill_id  result  shoot_type
# Conversion  reason
# GameStart
# MatchEnd
# GameEnd
# MentorsLog
# CheckAninTime
# ReserveAction  ?
# GameObInfo
# ShootCancel  result  skill_id  shoot_type



EncodeMap = {'GameStart': 1, 'GameEnd': 2, 'MatchEnd': 3, 'Pass': 4, 'Skill': 5, 'Shoot': 6, 'Conversion': 7, 'Block': 8, 'Steal': 9, 'Rebound': 10, 'MentorsLog': 11, 'CheckAninTime': 12, 'ReserveAction': 13, 'GameObInfo': 14,'ShootCancel':15}


def seq_gen_1(filepath,outpath):
    L=[]
    with open(filepath+'.txt','r') as f:
        datas = f.read().split('\n')
        for data in datas:
            for x in data.split(';'):
                result = x.split('@')[1]
                result = '1' if int(result)>0 else '-1'
                L.append(result+'@'+','.join([str(EncodeMap[action.split(':')[-1]]) for action in x.split('@')[2].split(',')]))

    with open(outpath+'.txt','w') as f:
        f.write('\n'.join(L))


def seq_gen_2(filepath,outpath):
    L=[]
    with open(filepath+'.txt','r') as f:
        datas = f.read().split('\n')
        for data in datas:
            tmp = []
            for x in data.split(';'):
                result = x.split('@')[1]
                result = '1' if int(result)>0 else '-1'
                for action in x.split('@')[2].split(','):
                    # print(action[:19])
                    tmp.append((action[:19],EncodeMap[action.split(':')[-1]]))
            sorted(tmp,key=lambda x : x[0])
            L.append(result+'@'+','.join([str(x[1]) for x in tmp]))

    with open(outpath+'.txt','w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    filepath = '../dataset/process_data/2018-11-21'
    outpath = '../dataset/ball/2018-11-21'
    seq_gen_2(filepath,outpath)


















