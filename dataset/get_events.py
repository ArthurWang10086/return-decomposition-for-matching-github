#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from log_utils import LogUtils
import requests
import json
from multiprocessing import Process, JoinableQueue
import multiprocessing
import datetime
import traceback

logger = LogUtils(__file__).get_logger()
# 数源接口
base_url = "http://42.186.114.228:8080/roleseq/time?game=ball&role_info=%s&start_time=%s 00:00:00&end_time=%s 23:59:59&event=none&dataNames=logid,origin_json"

OUTPUT = "behaviors_sql"
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

OUTPUT_Failed = "behaviors_sql/behaviors_seq_failed"
if not os.path.exists(OUTPUT_Failed):
    os.makedirs(OUTPUT_Failed)


def fetch_one_day(ds):
    if not os.path.exists(os.path.join(OUTPUT, ds)):
        os.makedirs(os.path.join(OUTPUT, ds))
    if not os.path.exists(os.path.join(OUTPUT_Failed, ds)):
        os.makedirs(os.path.join(OUTPUT_Failed, ds))
    role_ids = []
    with open('new_role_ids/%s.csv' % ds, 'r') as fin:
        for l in fin:
            for id in l.split(','):
                role_ids.append(id[:9])
    role_ids = list(set(role_ids))
    start_time = '%s' % ds
    _t = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    end_time = (_t + datetime.timedelta(days=1, seconds=-1)).strftime('%Y-%m-%d')
    for role_id in role_ids:
        url = base_url % (role_id, start_time, end_time)
        print(url)
        try:
            r = requests.post(url, timeout=600)
            json_dict = r.json()
            seq = json_dict[0]['role_seq']
            with open(os.path.join(OUTPUT, ds, '%s.json' % role_id), 'w') as f1:
                json.dump(seq, f1, indent=4, sort_keys=True)
            logger.debug("%s done.", role_id)
        except Exception as e:
            logger.error("%s, %s, %s", e, traceback.format_exc(), url)
            with open(os.path.join(OUTPUT_Failed, ds, str(role_id)), 'w') as f2:
                f2.write("%s, %s, %s\n" % (e, traceback.format_exc(), url))


def retry_failed():
    days = os.listdir(OUTPUT_Failed)
    for ds in days:
        failed_role_ids = os.listdir(os.path.join(OUTPUT_Failed, ds))
        for role_id in failed_role_ids:
            try:
                start_time = '%s' % ds
                _t = datetime.datetime.strptime(start_time, '%Y-%m-%d')
                end_time = (_t + datetime.timedelta(days=1, seconds=-1)).strftime('%Y-%m-%d')
                url = base_url % (role_id, start_time, end_time)
                try:
                    r = requests.post(url, timeout=600)
                    json_dict = r.json()
                    seq = json_dict[0]['role_seq']
                    with open(os.path.join(OUTPUT, ds, '%s.json' % role_id), 'w') as f1:
                        json.dump(seq, f1, indent=4, sort_keys=True)
                    os.remove(os.path.join(OUTPUT_Failed, ds, role_id))
                    logger.debug("%s done.", role_id)
                except Exception as e:
                    logger.error("%s, %s, %s", e, traceback.format_exc(), url)
            except Exception as e:
                logger.error("%s, %s", e, traceback.format_exc())


if __name__ == "__main__":
    DATE='2018-11-21'
    # create the pool
    pool = multiprocessing.Pool(processes=1)
    days = [DATE,]
    q = JoinableQueue()
    for ds in days:
        pool.apply_async(fetch_one_day, args=(ds, ))
    pool.close()
    pool.join()