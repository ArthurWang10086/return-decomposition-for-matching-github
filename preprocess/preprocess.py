#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Block   blocked_player  skill_id  result
# Rebound   result  skill_id  rebound_type
# Steal  result stealed_player skill_id
# Pass    receiver_id  skill_id(6 705 404 704 701 710 703)  pass_type(pass_short_nicepass pass_short intercept pass_long_run pass_long_lose)
# Skill  skill_id
# Shoot  skill_id(101 305 4 211 209 15 )  result  shoot_type
# Conversion  reason(pickup rebound)
# ShootCancel  result  skill_id  shoot_type


import json
import os
from multiprocessing import Process, JoinableQueue
import multiprocessing

# EncodeMap = {'Pass': 1, 'Skill': 2, 'Shoot': 3, 'Conversion': 4, 'Block': 5, 'Steal': 6, 'Rebound': 7,'ShootCancel':8}


def extraction_info(info):
    id = info['log_id']
    if id=='Pass':
        # return '#'.join([str(x) for x in [id,info['receiver_id'],info['skill_id'],info['pass_type']]])
        return '#'.join([str(x) for x in [id,info['skill_id'],info['pass_type']]])
    elif id == 'Skill':
        return '#'.join([str(x) for x in [id,info['skill_id']]])
    elif id == 'Shoot':
        return '#'.join([str(x) for x in [id,info['skill_id'],info['result'],info['shoot_type']]])
    elif id == 'Conversion':
        return '#'.join([str(x) for x in [id,info['reason']]])
    elif id == 'Block':
        # return '#'.join([str(x) for x in [id,info['blocked_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['skill_id'],info['result']]])
    elif id == 'Steal':
        # return '#'.join([str(x) for x in [id,info['stealed_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['skill_id'],info['result']]])
    elif id == 'Rebound':
        return '#'.join([str(x) for x in [id,info['rebound_type'],info['skill_id'],info['result']]])
    elif id == 'ShootCancel':
        return '#'.join([str(x) for x in [id,info['result'],info['skill_id'],info['shoot_type']]])
    else:
        return ''


def process(ds):
    D = {}
    filepath = '../dataset/behaviors_sql/%s'%(ds)
    outpath = '../dataset/process_data/%s'%(ds)
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
    # import copy
    # D1 = copy.deepcopy(D)
    # for gameid in D:
    #     for role_id in D[gameid]:
    #         GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
    #         GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
    #         game_result = max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
    #         if GameEnd_count!=1 or GameStart_count!=1 or game_result not in [0,1]:
    #             D1[gameid].pop(role_id)

    #清洗比赛
    print(ds,len(D))
    L = filter(lambda x:len(D[x])==6 ,D.keys())
    print(ds,len(L))

    D2={}
    for gameid in L:
        D2[gameid] = []
        for role_id in D[gameid]:
            # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
            if game_result == '1':
                D2[gameid].append(role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id']+':'+extraction_info(json.loads(x)) for x in D[gameid][role_id]]))

        for role_id in D[gameid]:
            # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
            if game_result == '0':
                D2[gameid].append(role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id']+':'+extraction_info(json.loads(x)) for x in D[gameid][role_id]]))

    with open(outpath+'.txt','w') as f:
        L = [';'.join(D2[x]) for x in D2 ]
        f.write('\n'.join(L))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=5)
    days = ['2018-11-01','2018-11-02','2018-11-03','2018-11-04','2018-11-05']
    q = JoinableQueue()
    for ds in days:
        pool.apply_async(process, args=(ds, ))
    pool.close()
    pool.join()











