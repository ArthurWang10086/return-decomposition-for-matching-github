#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Block
# Rebound
# Steal
# Pass
# Skill
# Shoot
# Conversion
# GameStart
# MatchEnd
# GameEnd
# MentorsLog
# CheckAninTime
# ReserveAction
# GameObInfo


import json
import os

if __name__ == '__main__':
    D = {}
    D1 = {}
    filepath = '../dataset/behaviors_sql/2018-11-21'
    outpath = '../dataset/process_data/2018-11-21'
    role_ids = filter(lambda x:len(x)==9,[x.split('.')[0] for x in os.listdir(filepath)])
    for role_id in role_ids:
        with open(filepath+'/%s.json'%(role_id),'r') as f:
            try:
                datas = json.loads(f.read())
                for data in datas :
                    if 'game_uuid' in data[0]['origin_json']:
                        gameid = json.loads(data[0]['origin_json'])['game_uuid']
                        if gameid in D:
                            if role_id in D[gameid] :
                                D[gameid][role_id].append(data[0]['origin_json'])
                            else:
                                D[gameid][role_id] = [data[0]['origin_json']]
                        else:
                            D[gameid] = {}
                            D[gameid][role_id] = [data[0]['origin_json']]
            except Exception:
                pass
    #清洗序列
    import copy
    D1 = copy.deepcopy(D)
    for gameid in D:
        for role_id in D[gameid]:
            GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            # game_result = max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            if GameEnd_count!=1 or GameStart_count!=1:
                D1[gameid].pop(role_id)

    #清洗比赛
    L = filter(lambda x:len(D[x])==6 ,D.keys())

    D2={}
    for gameid in L:
        D2[gameid] = []
        for role_id in D[gameid]:
            # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
            if game_result == '1':
                D2[gameid].append(role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id'] for x in D[gameid][role_id]]))

    for gameid in L:
        D2[gameid] = []
        for role_id in D[gameid]:
            # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
            if game_result == '0':
                D2[gameid].append(role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id'] for x in D[gameid][role_id]]))

    with open(outpath+'.txt','w') as f:
        L = [';'.join(D2[x]) for x in D2 ]
        f.write('\n'.join(L))








