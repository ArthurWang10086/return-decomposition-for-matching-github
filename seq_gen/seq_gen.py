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


def seq_gen_1(ds):
    filepath = '../dataset/process_data/%s'%(ds)
    outpath = '../dataset/ball/%s'%(ds)
    L=[]
    EncodeMap = actionEncode.EncodeMap
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
    EncodeMap = actionEncode.EncodeMap
    EncodeMap2 = actionEncode.EncodeMap2
    f = open(filepath+'.txt','r')
    datas = f.read().split('\n')
    for data in datas:
        tmp = []
        for i,x in enumerate(data.split(';')):
            result = x.split('@')[1]
            result = '1' if int(result)>0 else '-1'
            for action in x.split('@')[2].split(','):
                if len(action.split(':')[-1])>0:
                    tmp.append((action[:19],str(i)+':'+action.split(':')[-1]))
        sorted(tmp,key=lambda x : x[0])
        if len(tmp)>1:
            L.append('1'+'@'+','.join([str(x[1]) for x in tmp]))

    with open(outpath+'.txt','w') as f:
        f.write('\n'.join(L))

def seq_gen_3(ds):
    filepath = '../dataset/process_data/%s'%(ds)
    outpath = '../dataset/ball/%s'%(ds)
    L=[]
    EncodeMap = actionEncode.EncodeMap
    EncodeMap2 = actionEncode.EncodeMap2
    f = open(filepath+'.txt','r')
    datas = f.read().split('\n')
    for data in datas:
        tmp = []
        scores=[]
        for i,x in enumerate(data.split(';')):
            score=0
            result = x.split('@')[1]
            result = '1' if int(result)>0 else '-1'
            for action in x.split('@')[2].split(','):
                if len(action.split(':')[-1])>0 and ('Conversion#None#goalin#None' in action.split(':')[-1]):
                    playerid = 0 if i <3 else 5
                    tmp.append((action[:19],str(playerid)+':'+action.split(':')[-1]))
                    score += 1 if 'Conversion#None#goalin#None' in action.split(':')[-1] else 0
            scores.append(score)
        tmp = sorted(tmp,key=lambda x : x[0])
        diffscore = sum(scores[3:])-sum(scores[:3])
        # diffscore = sum(scores)
        if len(tmp)>1:
            L.append(str(diffscore)+'@'+','.join([str(x[1]) for x in tmp]))
        tmp = []
        scores=[]
        for i,x in enumerate(data.split(';')):
            score=0
            result = x.split('@')[1]
            result = '1' if int(result)>0 else '-1'
            for action in x.split('@')[2].split(','):
                if len(action.split(':')[-1])>0 and ('Conversion#None#goalin#None' in action.split(':')[-1]):
                    playerid = 0 if i <3 else 5
                    tmp.append((action[:19],str(5-playerid)+':'+action.split(':')[-1]))
                    score += 1 if 'Conversion#None#goalin#None' in action.split(':')[-1] else 0
            scores.append(score)
        tmp = sorted(tmp,key=lambda x : x[0])
        diffscore = -(sum(scores[3:])-sum(scores[:3]))
        if len(tmp)>1:
            L.append(str(diffscore)+'@'+','.join([str(x[1]) for x in tmp]))


        # tmp = []
        # scores=[]
        # for i,x in enumerate(data.split(';')):
        #     score=0
        #     result = x.split('@')[1]
        #     result = '1' if int(result)>0 else '-1'
        #     for action in x.split('@')[2].split(','):
        #         if len(action.split(':')[-1])>0 and ('Conversion' in action.split(':')[-1]):
        #             playerid = 0 if i <3 else 5
        #             tmp.append((action[:19],str(playerid)+':'+action.split(':')[-1]))
        #             score += 1 if 'Conversion#None#goalin#None' in action.split(':')[-1] else 0
        #     scores.append(score)
        # tmp = sorted(tmp,key=lambda x : x[0])
        # diffscore = sum(scores[3:])-sum(scores[:3])
        # # diffscore = sum(scores)
        # if len(tmp)>1:
        #     L.append(str(diffscore)+'@'+','.join([str(x[1]) for x in tmp]))
        # tmp = []
        # scores=[]
        # for i,x in enumerate(data.split(';')):
        #     score=0
        #     result = x.split('@')[1]
        #     result = '1' if int(result)>0 else '-1'
        #     for action in x.split('@')[2].split(','):
        #         if len(action.split(':')[-1])>0 and ('Conversion' in action.split(':')[-1]):
        #             playerid = 0 if i <3 else 5
        #             tmp.append((action[:19],str(5-playerid)+':'+action.split(':')[-1]))
        #             score += 1 if 'Conversion#None#goalin#None' in action.split(':')[-1] else 0
        #     scores.append(score)
        # tmp = sorted(tmp,key=lambda x : x[0])
        # diffscore = -(sum(scores[3:])-sum(scores[:3]))
        # if len(tmp)>1:
        #     L.append(str(diffscore)+'@'+','.join([str(x[1]) for x in tmp]))

    with open(outpath+'.diff.txt','w') as f:
        f.write('\n'.join(L[:1000]))

if __name__ == '__main__':
    seq_gen_3('2018-11-01')
    # pool = multiprocessing.Pool(processes=1)
    # days = ['2018-11-01']
    # q = JoinableQueue()
    # for ds in days:
    #     pool.apply_async(seq_gen_2, args=(ds, ))
    # pool.close()
    # pool.join()




















