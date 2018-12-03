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

EncodeMap = {'Pass#203124937#6#pass_short': 1, 'Steal#103656190#8#fail': 3, 'Pass#200720126#6#pass_short': 145, 'Pass#102137991#6#pass_short': 4, 'Block#202711721#9#fail': 5, 'Block#202718448#605#fail': 6, 'Pass#202607635#6#pass_short': 7, 'Shoot#201#nogoal#middle': 8, 'Pass#100114179#705#pass_short_nicepass': 9, 'Shoot#15#nogoal#layupndunk': 10, 'Conversion#pickup': 11, 'ShootCancel#nogoal#211#three_point': 12, 'Block#200532852#9#fail': 13, 'Shoot#211#nogoal#three_point': 14, 'Conversion#goalin': 15, 'Pass#202334874#404#pass_short': 16, 'Pass#202803359#6#pass_short': 17, 'Rebound#normal#1#success': 18, 'Steal#103109298#8#fail': 199, 'Skill#704': 191, 'Block#101614300#9#fail': 19, 'Shoot#212#nogoal#three_point': 68, 'Block#100219496#9#fail': 21, 'Shoot#202#nogoal#near': 22, 'Steal#100114415#8#fail': 23, 'Skill#705': 192, 'Block#100114708#605#fail': 24, 'Pass#202334874#6#pass_short': 25, 'Pass#100114601#6#pass_short': 27, 'Shoot#107#nogoal#layupndunk': 28, 'Pass#100114110#6#pass_short': 29, 'Pass#202607635#6#pass_short_nicepass': 198, 'Block#200123059#9#success': 30, 'Block#100114661#9#fail': 31, 'Block#200627970#9#fail': 32, 'Pass#101614300#6#pass_short': 33, 'Rebound#normal#1#fail': 34, 'Block#202718448#9#success': 35, 'Skill#605': 36, 'Steal#202711721#501#success': 26, 'Pass#200532487#705#pass_short': 38, 'Skill#312': 39, 'Skill#603': 40, 'Skill#602': 41, 'Skill#4': 42, 'Pass#100114601#404#pass_short': 44, 'Skill#103': 159, 'Pass#100114110#404#pass_short_nicepass': 45, 'Pass#200532487#404#intercept': 46, 'Steal#102825020#505#success': 47, 'Pass#100114631#704#pass_short': 48, 'Steal#102825020#505#fail': 49, 'Shoot#101#nogoal#layupndunk': 181, 'Pass#201022929#6#pass_short_nicepass': 51, 'Steal#200140963#8#success': 0, 'Skill#308': 53, 'Shoot#211#nogoal#middle': 153, 'Pass#100120556#6#pass_short': 55, 'Skill#301': 56, 'Block#100115096#9#success': 57, 'Skill#111': 208, 'Skill#304': 58, 'Steal#202711721#8#fail': 59, 'Pass#102926706#6#pass_short': 60, 'Block#102137991#9#fail': 196, 'Shoot#205#nogoal#near': 62, 'Steal#103656190#505#fail': 63, 'Block#200229251#605#fail': 64, 'Steal#100114129#8#fail': 212, 'Block#100114323#9#fail': 61, 'Pass#100115635#6#pass_short': 67, 'Pass#100114631#6#pass_short': 69, 'Pass#200532852#404#pass_short': 37, 'Pass#100124838#6#intercept': 71, 'Conversion#clock_violation': 72, 'Pass#202711721#6#pass_short': 73, 'Pass#100114601#6#pass_short_nicepass': 74, 'Pass#202607635#6#pass_long_run': 75, 'Block#202718448#9#fail': 76, 'ShootCancel#nogoal#108#layupndunk': 77, 'Block#103109298#9#fail': 78, 'Skill#505': 79, 'Steal#103027923#505#fail': 80, 'Block#103656190#9#success': 81, 'Skill#501': 82, 'ShootCancel#nogoal#4#middle': 83, 'Skill#202': 84, 'Conversion#block_catch': 85, 'Skill#701': 195, 'Skill#205': 120, 'Pass#103109298#6#pass_short': 89, 'Pass#100120556#404#pass_short': 90, 'Block#100124838#9#fail': 91, 'ShootCancel#nogoal#116#three_point': 92, 'Block#201726665#9#fail': 93, 'Pass#202426398#404#pass_short': 194, 'Pass#200726081#6#pass_short': 94, 'ShootCancel#nogoal#4#three_point': 95, 'Skill#107': 96, 'Pass#201022929#404#pass_short': 97, 'Pass#200916494#701#intercept': 98, 'Steal#202718448#501#fail': 99, 'Block#202627293#9#success': 100, 'Skill#21': 101, 'Skill#23': 102, 'Pass#200532487#6#pass_short': 103, 'Block#202426398#9#fail': 104, 'Pass#100124838#404#pass_short': 105, 'ShootCancel#nogoal#101#layupndunk': 106, 'Pass#100114178#705#pass_short': 107, 'Pass#200323366#404#pass_short': 108, 'Block#101614300#605#fail': 109, 'Pass#200229251#6#pass_short': 110, 'Block#201715225#9#fail': 111, 'Pass#202426398#6#pass_short': 112, 'Block#200325218#9#fail': 113, 'Shoot#111#nogoal#layupndunk': 114, 'Steal#200229251#8#fail': 115, 'Block#100114708#9#fail': 124, 'Skill#201': 117, 'Steal#100115096#8#fail': 118, 'Steal#202334874#8#fail': 65, 'Steal#202917140#8#fail': 88, 'Rebound#block_highball#1#fail': 121, 'Skill#209': 123, 'Pass#202939238#404#pass_short': 116, 'Block#201521202#9#success': 125, 'Pass#100114631#6#pass_short_nicepass': 126, 'Pass#102926706#404#pass_short': 20, 'Shoot#4#nogoal#three_point': 129, 'Block#100115096#9#fail': 130, 'Pass#103109298#404#pass_short': 86, 'Pass#201022929#6#pass_short': 132, 'Skill#329': 122, 'Steal#201726665#501#success': 135, 'Steal#102825020#8#fail': 136, 'Skill#34': 137, 'ShootCancel#nogoal#225#three_point': 138, 'Shoot#209#nogoal#middle': 139, 'Steal#102137991#8#fail': 52, 'Steal#201022929#8#fail': 141, 'Block#202627293#9#fail': 142, 'Skill#404': 143, 'Skill#401': 144, 'Pass#100114661#6#pass_short': 127, 'Pass#202607635#704#pass_short': 146, 'Pass#202419180#404#pass_short': 148, 'Block#202939238#605#fail': 161, 'Skill#212': 149, 'Skill#211': 150, 'Pass#200532487#701#pass_short': 151, 'Pass#200916494#701#pass_short': 152, 'Conversion#rebound': 54, 'Pass#101614300#6#pass_short_nicepass': 154, 'Pass#203347260#6#pass_short': 155, 'Steal#200627970#8#fail': 156, 'Block#201521202#9#fail': 157, 'Block#200627970#605#fail': 131, 'Pass#202939238#6#pass_short': 134, 'Pass#200726081#404#pass_short': 160, 'Pass#100114179#705#pass_short': 70, 'Pass#100114601#404#intercept': 162, 'Block#103656190#9#fail': 163, 'ShootCancel#goalin#4#three_point': 164, 'Block#200123059#9#fail': 187, 'Shoot#103#nogoal#layupndunk': 165, 'Pass#203240227#6#pass_short': 166, 'Block#200916494#605#fail': 167, 'Pass#101614300#404#pass_short': 168, 'Block#100115096#602#success': 169, 'Shoot#4#nogoal#near': 119, 'Steal#101614300#501#fail': 170, 'Pass#100115635#404#pass_short': 171, 'Block#202939238#9#fail': 172, 'Shoot#4#nogoal#middle': 147, 'Pass#101737725#6#pass_short': 43, 'Steal#201321668#8#fail': 175, 'Block#201521202#605#success': 176, 'Steal#103656190#501#fail': 177, 'ShootCancel#nogoal#103#layupndunk': 178, 'Steal#202627293#8#fail': 179, 'ShootCancel#nogoal#107#layupndunk': 50, 'Block#200140963#9#fail': 182, 'Pass#100114179#6#pass_short': 183, 'Pass#202803359#6#pass_long_run': 184, 'Block#103025175#9#fail': 185, 'ShootCancel#nogoal#211#middle': 186, 'Pass#200532852#6#pass_short': 66, 'Block#201521202#605#fail': 188, 'Pass#100114661#6#pass_short_nicepass': 189, 'Pass#200627970#6#pass_short': 190, 'Steal#203240227#8#success': 2, 'Pass#100124838#6#pass_short': 173, 'Steal#200325218#8#fail': 193, 'Block#202803359#9#success': 140, 'Steal#202711721#501#fail': 128, 'Block#100115096#602#fail': 174, 'ShootCancel#nogoal#212#three_point': 133, 'Skill#330': 197, 'Block#202221845#9#fail': 180, 'Pass#200916494#705#pass_short': 158, 'Pass#100114110#404#pass_short': 200, 'Block#100114323#603#fail': 201, 'Conversion#steal_catch': 202, 'Pass#200532487#404#pass_short_nicepass': 203, 'Pass#200532487#404#pass_short': 204, 'ShootCancel#nogoal#202#near': 205, 'Pass#100114178#6#pass_short': 206, 'Block#201022929#9#fail': 207, 'Steal#100115096#8#success': 87, 'Pass#100114179#701#pass_short': 209, 'Block#202702683#9#fail': 210, 'Pass#200532487#705#pass_short_nicepass': 211, 'Pass#202419180#6#pass_short': 213, 'Shoot#15#nogoal#near': 214}


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




















