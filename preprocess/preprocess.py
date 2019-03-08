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

import traceback
import simplejson as json
import os
from multiprocessing import Process, JoinableQueue
import multiprocessing
from base64 import b64decode
from base64 import b64encode

# EncodeMap = {'Pass': 1, 'Skill': 2, 'Shoot': 3, 'Conversion': 4, 'Block': 5, 'Steal': 6, 'Rebound': 7,'ShootCancel':8}


def extraction_info(info):
    id = info['log_id']
    if id=='Pass':
        # return '#'.join([str(x) for x in [id,info['receiver_id'],info['skill_id'],info['pass_type']]])
        return '#'.join([str(x) for x in [id,'None',info['pass_type'],info['skill_id']]])
    elif id == 'Skill':
        return '#'.join([str(x) for x in [id,'None','None',info['skill_id']]])
    # elif id == 'Shoot':
    #     return '#'.join([str(x) for x in [id,info['result'],info['shoot_type'],info['skill_id']]])
    elif id == 'Conversion':
        return '#'.join([str(x) for x in [id,'None',info['reason'],'None']])
    elif id == 'Block':
        # return '#'.join([str(x) for x in [id,info['blocked_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['result'],'None',info['skill_id']]])
    elif id == 'Steal':
        # return '#'.join([str(x) for x in [id,info['stealed_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['result'],'None',info['skill_id']]])
    elif id == 'Rebound':
        return '#'.join([str(x) for x in [id,info['result'],info['rebound_type'],info['skill_id']]])
    elif id == 'ShootCancel':
        return '#'.join([str(x) for x in [id,info['result'],info['shoot_type'],info['skill_id']]])
    elif id == 'ShootResult':
        return '#'.join([str(x) for x in [id,info['score'],info['shoot_type'],'None']])
    else:
        return id+'#None#None#None'

def extraction_info2(info):
    id = info['log_id']
    if id=='Pass':
        # return '#'.join([str(x) for x in [id,info['receiver_id'],info['skill_id'],info['pass_type']]])
        return '#'.join([str(x) for x in [id,'None',info['pass_type'],info['skill_id'],info['game_remain_time']]])
    elif id == 'Skill':
        return '#'.join([str(x) for x in [id,'None','None',info['skill_id'],info['game_remain_time']]])
    elif id == 'Conversion':
        return '#'.join([str(x) for x in [id,'None',info['reason'],info['game_score'].replace(':','-'),info['game_remain_time']]])
    elif id == 'Block':
        # return '#'.join([str(x) for x in [id,info['blocked_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['result'],'None',info['skill_id'],info['game_remain_time']]])
    elif id == 'Steal':
        # return '#'.join([str(x) for x in [id,info['stealed_player'],info['skill_id'],info['result']]])
        return '#'.join([str(x) for x in [id,info['result'],'None',info['skill_id'],info['game_remain_time']]])
    elif id == 'Rebound':
        return '#'.join([str(x) for x in [id,info['result'],info['rebound_type'],info['skill_id'],info['game_remain_time']]])
    elif id == 'ShootResult':
        return '#'.join([str(x) for x in [id,info['score'],info['shoot_type'],'None',info['game_remain_time']]])
    elif id == 'GameEnd':
        #json.loads(b64decode(x).decode())
        return id+'#'+b64encode(json.dumps(info).encode('utf-8')).decode()+'#'+str(info['role_score'])+'#'\
               +str(info['game_score']).replace(':','-')+'#None'
    else:
        return id+'#None#None#None'

def process(ds):
    D = {}
    filepath = '../dataset/behaviors_sql/%s'%(ds)
    outpath = '../dataset/process_data/%s'%(ds)
    role_ids = list(filter(lambda x:len(x)==9,[x.split('.')[0] for x in os.listdir(filepath)]))
    for role_id in role_ids:
        with open(filepath+'/%s.json'%(role_id),'r') as f:
            try:
                datas = json.loads(f.read())
                for data in datas :
                    if 'origin_json' in data[0] and 'game_uuid' in data[0]['origin_json'] :
                        gameid = json.loads(data[0]['origin_json'])['game_uuid']
                        if gameid in D:
                            if role_id in D[gameid] :
                                D[gameid][role_id].append(data[0]['origin_json'])
                            else:
                                D[gameid][role_id] = [data[0]['origin_json']]
                        else:
                            D[gameid] = {}
                            D[gameid][role_id] = [data[0]['origin_json']]
            except Exception as  e:
                print(role_id)
                traceback.print_exc()
                pass
    #清洗序列
    import copy
    D1 = copy.deepcopy(D)
    for gameid in D:
        for role_id in D[gameid]:
            GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
            game_result = max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            game_type = max([int(json.loads(x)['game_type']) if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
            if GameEnd_count!=1 or GameStart_count!=1 or game_result not in [0,1] or game_type!=2 :
                print(gameid,GameEnd_count,GameStart_count,game_result)
                D1[gameid].pop(role_id)

    #清洗比赛
    print(ds,len(D))
    # print(D['d369b4f10917e08c873935b53773c3ab'])
    D = D1
    # json.dump(D, open(ds+'.dict', "w"))
    L = list(filter(lambda x:len(D[x])==6 ,D.keys()))
    print(list(filter(lambda x:len(D[x])!=6 ,D.keys()))[:10])
    print(ds,len(L))

    D2={}
    for gameid in L:
        D2[gameid] = []
        try:
            for role_id in D[gameid]:
                # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
                # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
                game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
                if game_result == '1':
                    D2[gameid].append(gameid+'|'+role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id']+':'+extraction_info2(json.loads(x)) for x in D[gameid][role_id]]))

            for role_id in D[gameid]:
                # GameEnd_count = sum([1 if 'GameEnd' in x else 0 for x in D[gameid][role_id]])
                # GameStart_count = sum([1 if 'GameStart' in x else 0 for x in D[gameid][role_id]])
                game_result = str(max([json.loads(x)['game_result'] if 'GameEnd' in x else 0 for x in D[gameid][role_id]]))
                if game_result == '0':
                    D2[gameid].append(gameid+'|'+role_id+'@'+game_result+'@'+','.join([json.loads(x)['log_ts']+':'+json.loads(x)['log_id']+':'+extraction_info2(json.loads(x)) for x in D[gameid][role_id]]))
        except Exception as e:
            D2.pop(gameid)
            traceback.print_exc()
            pass

    print(len(D2))

    with open(outpath+'.txt','w') as f:
        L = [';'.join(D2[x]) for x in D2]
        f.write('\n'.join(L))


if __name__ == '__main__':
    process('2019-03-02')
    # pool = multiprocessing.Pool(processes=1)
    # days = ['2019-03-02']
    # q = JoinableQueue()
    # for ds in days:
    #     pool.apply_async(process, args=(ds, ))
    # pool.close()
    # pool.join()











