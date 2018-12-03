#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import preprocess
from multiprocessing import Process, JoinableQueue
import multiprocessing
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

EncodeMap = {'Skill#605': 0, 'Skill#211': 1, 'Steal#505#fail': 2, 'Skill#202': 3, 'Pass#6#pass_short_nicepass': 4, 'Skill#212': 5, 'Pass#705#pass_short': 6, 'Pass#404#intercept': 7, 'Rebound#block_highball#1#fail': 8, 'Pass#704#pass_short': 9, 'ShootCancel#goalin#4#three_point': 10, 'ShootCancel#nogoal#211#middle': 11, 'Pass#705#pass_short_nicepass': 12, 'Skill#705': 13, 'Pass#701#pass_short': 14, 'Conversion#clock_violation': 15, 'Pass#6#pass_long_run': 16, 'ShootCancel#nogoal#107#layupndunk': 17, 'ShootCancel#nogoal#225#three_point': 18, 'Rebound#normal#1#fail': 19, 'Rebound#normal#1#success': 20, 'Block#602#success': 21, 'Skill#23': 22, 'Skill#401': 23, 'Shoot#209#nogoal#middle': 24, 'ShootCancel#nogoal#211#three_point': 25, 'Shoot#15#nogoal#near': 26, 'Steal#8#fail': 27, 'Shoot#201#nogoal#middle': 28, 'Skill#201': 29, 'Skill#21': 30, 'Shoot#103#nogoal#layupndunk': 31, 'Skill#301': 32, 'Block#9#fail': 33, 'Shoot#4#nogoal#middle': 34, 'Skill#505': 35, 'ShootCancel#nogoal#103#layupndunk': 36, 'Shoot#211#nogoal#middle': 37, 'Pass#404#pass_short': 38, 'Shoot#211#nogoal#three_point': 39, 'Skill#304': 40, 'Shoot#202#nogoal#near': 41, 'Steal#501#fail': 42, 'Shoot#111#nogoal#layupndunk': 43, 'Shoot#212#nogoal#three_point': 44, 'Conversion#pickup': 45, 'ShootCancel#nogoal#202#near': 46, 'Conversion#block_catch': 47, 'Skill#209': 48, 'ShootCancel#nogoal#116#three_point': 49, 'Skill#701': 50, 'ShootCancel#nogoal#4#three_point': 51, 'Pass#6#pass_short': 52, 'ShootCancel#nogoal#101#layupndunk': 53, 'Block#602#fail': 54, 'Skill#329': 55, 'Steal#8#success': 56, 'Skill#312': 57, 'Skill#308': 58, 'Skill#4': 59, 'Skill#501': 60, 'Block#605#fail': 61, 'Skill#111': 62, 'Block#9#success': 63, 'Shoot#15#nogoal#layupndunk': 64, 'Skill#34': 65, 'Skill#603': 66, 'Skill#404': 67, 'Steal#505#success': 68, 'ShootCancel#nogoal#108#layupndunk': 69, 'Shoot#4#nogoal#near': 70, 'Skill#330': 71, 'Skill#602': 72, 'Pass#6#intercept': 73, 'ShootCancel#nogoal#212#three_point': 74, 'Conversion#rebound': 75, 'Steal#501#success': 76, 'Block#605#success': 77, 'Pass#701#intercept': 78, 'Shoot#101#nogoal#layupndunk': 79, 'Shoot#205#nogoal#near': 80, 'Conversion#steal_catch': 81, 'Block#603#fail': 82, 'Skill#205': 83, 'Shoot#4#nogoal#three_point': 84, 'Conversion#goalin': 85, 'Skill#103': 86, 'ShootCancel#nogoal#4#middle': 87, 'Skill#704': 88, 'Shoot#107#nogoal#layupndunk': 89, 'Skill#107': 90, 'Pass#404#pass_short_nicepass': 91}


def actionEncode(days):
    actions=set()
    for ds in days:
        filepath = '../dataset/process_data/%s'%(ds)
        outpath = '../dataset/ball/%s'%(ds)
        with open(filepath+'.txt','r') as f:
            datas = f.read().split('\n')
            for data in datas:
                for x in data.split(';'):
                    for action in x.split('@')[2].split(','):
                        if len(action.split(':')[-1])>0:
                            actions.add(action.split(':')[-1])
    L=list(actions)
    return str(dict(zip(L,range(len(actions)))))


if __name__ == '__main__':
    result = actionEncode(['2018-11-01','2018-11-02','2018-11-03','2018-11-04','2018-11-05'])
    print(result)




















