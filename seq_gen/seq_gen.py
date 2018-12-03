#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from multiprocessing import Process, JoinableQueue
import multiprocessing
import  actionEncode
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

EncodeMap = actionEncode.EncodeMap

def seq_gen_1(ds):
    filepath = '../dataset/process_data/%s'%(ds)
    outpath = '../dataset/ball/%s'%(ds)
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


def seq_gen_2(ds):
    filepath = '../dataset/process_data/%s'%(ds)
    outpath = '../dataset/ball/%s'%(ds)
    L=[]
    with open(filepath+'.txt','r') as f:
        datas = f.read().split('\n')
        for data in datas:
            tmp = []
            for i,x in enumerate(data.split(';')):
                result = x.split('@')[1]
                result = '1' if int(result)>0 else '-1'
                for action in x.split('@')[2].split(','):
                    # print(action[:19])
                    #action_id = logid + role_idx * len(logids)
                    if action.split(':')[-1] in EncodeMap:
                        tmp.append((action[:19],EncodeMap[action.split(':')[-1]]+i*len(EncodeMap)))
            sorted(tmp,key=lambda x : x[0])
            L.append('1'+'@'+','.join([str(x[1]) for x in tmp]))

    with open(outpath+'.txt','w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=1)
    days = ['2018-11-01','2018-11-02','2018-11-03','2018-11-04','2018-11-05']
    q = JoinableQueue()
    for ds in days:
        pool.apply_async(seq_gen_2, args=(ds, ))
    pool.close()
    pool.join()




















