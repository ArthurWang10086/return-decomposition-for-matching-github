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
# EncodeMap = {'Conversion#None#goalin#None': 0, 'Conversion#None#steal_catch#None': 1, 'Conversion#None#clock_violation#None': 2, 'Conversion#None#block_catch#None': 3, 'Conversion#None#pickup#None': 4, 'Conversion#None#BuzzerBeater#None': 5, 'Conversion#None#rebound#None': 6}
pass_short pass_short_nicepass intercept intercept_catch pass_long_lose pass_long_run

EncodeMap = {'ShootCancel#goalin#layupndunk#15': 0, 'Rebound#success#normal#402': 1, 'Shoot#nogoal#three_point#222': 2, 'Rebound#fail#normal#112': 3, 'Shoot#goalin#layupndunk#108': 4, 'Skill#None#None#103': 5, 'Pass#None#pass_short#701': 6, 'Shoot#nogoal#three_point#215': 7, 'ShootCancel#nogoal#three_point#214': 8, 'Skill#None#None#109': 9, 'Skill#None#None#329': 10, 'ShootCancel#goalin#middle#209': 11, 'ShootCancel#goalin#three_point#227': 12, 'Shoot#goalin#middle#29': 13, 'ShootCancel#nogoal#near#208': 14, 'Skill#None#None#108': 15, 'Pass#None#pass_short_nicepass#710': 16, 'Rebound#success#block_highball#1': 17, 'Shoot#goalin#middle#224': 18, 'Skill#None#None#4': 19, 'Shoot#goalin#middle#209': 20, 'Shoot#goalin#middle#213': 21, 'Skill#None#None#505': 22, 'Shoot#goalin#near#325': 23, 'ShootCancel#goalin#middle#228': 24, 'Skill#None#None#209': 25, 'Shoot#nogoal#three_point#217': 26, 'Shoot#goalin#middle#702': 27, 'ShootCancel#nogoal#layupndunk#202': 28, 'Shoot#nogoal#near#205': 29, 'Shoot#nogoal#layupndunk#202': 30, 'Shoot#nogoal#layupndunk#207': 31, 'Skill#None#None#34': 32, 'Shoot#nogoal#near#29': 33, 'Shoot#nogoal#three_point#116': 34, 'ShootCancel#nogoal#middle#203': 35, 'ShootCancel#nogoal#middle#702': 36, 'ShootCancel#nogoal#near#211': 37, 'Shoot#nogoal#layupndunk#108': 38, 'Shoot#nogoal#layupndunk#111': 39, 'Shoot#goalin#three_point#225': 40, 'ShootCancel#nogoal#layupndunk#114': 41, 'ShootCancel#nogoal#layupndunk#210': 42, 'Shoot#nogoal#middle#15': 43, 'ShootCancel#nogoal#near#210': 44, 'Skill#None#None#703': 45, 'Rebound#fail#normal#1': 46, 'Pass#None#intercept#703': 47, 'ShootCancel#goalin#three_point#212': 48, 'Pass#None#intercept#704': 49, 'Shoot#goalin#three_point#213': 50, 'ShootCancel#nogoal#three_point#229': 51, 'ShootCancel#nogoal#three_point#211': 52, 'ShootCancel#goalin#layupndunk#101': 53, 'Skill#None#None#404': 54, 'ShootCancel#goalin#three_point#4': 55, 'ShootCancel#goalin#middle#213': 56, 'Skill#None#None#224': 57, 'Skill#None#None#229': 58, 'Shoot#nogoal#middle#218': 59, 'Skill#None#None#228': 60, 'Skill#None#None#217': 61, 'Shoot#nogoal#near#702': 62, 'ShootCancel#nogoal#three_point#218': 63, 'Steal#fail#None#501': 64, 'Steal#fail#None#505': 65, 'Shoot#nogoal#three_point#218': 66, 'Skill#None#None#112': 67, 'ShootCancel#nogoal#layupndunk#102': 68, 'ShootCancel#goalin#middle#217': 69, 'ShootCancel#nogoal#near#218': 70, 'Shoot#nogoal#layupndunk#101': 71, 'Skill#None#None#704': 72, 'Shoot#nogoal#near#210': 73, 'Shoot#goalin#near#208': 74, 'Shoot#nogoal#near#218': 75, 'Pass#None#intercept#6': 76, 'Shoot#goalin#three_point#203': 77, 'Block#fail#None#9': 78, 'Steal#success#None#501': 79, 'ShootCancel#nogoal#middle#325': 80, 'Shoot#nogoal#middle#205': 81, 'Skill#None#None#222': 82, 'ShootCancel#nogoal#near#202': 83, 'Skill#None#None#308': 84, 'ShootCancel#nogoal#middle#212': 85, 'Skill#None#None#218': 86, 'Conversion#None#goalin#None': 87, 'ShootCancel#nogoal#three_point#227': 88, 'Skill#None#None#324': 89, 'ShootCancel#nogoal#middle#211': 90, 'Skill#None#None#330': 91, 'ShootCancel#nogoal#middle#205': 92, 'Skill#None#None#328': 93, 'Shoot#nogoal#three_point#227': 94, 'Conversion#None#steal_catch#None': 95, 'Shoot#nogoal#near#202': 96, 'Shoot#nogoal#near#208': 97, 'Pass#None#intercept_catch#704': 98, 'Shoot#goalin#layupndunk#15': 99, 'Shoot#nogoal#three_point#211': 100, 'Shoot#nogoal#middle#212': 101, 'Shoot#goalin#three_point#4': 102, 'Pass#None#intercept#701': 103, 'ShootCancel#nogoal#near#325': 104, 'Skill#None#None#203': 105, 'Rebound#fail#block_heighball#112': 106, 'ShootCancel#goalin#middle#218': 107, 'Pass#None#pass_long_lose#6': 108, 'Skill#None#None#602': 109, 'Shoot#nogoal#three_point#213': 110, 'Shoot#nogoal#middle#29': 111, 'Rebound#fail#block_highball#402': 112, 'Rebound#success#normal#112': 113, 'Pass#None#pass_short#704': 114, 'Conversion#None#clock_violation#None': 115, 'Shoot#goalin#three_point#116': 116, 'Conversion#None#block_catch#None': 117, 'Skill#None#None#402': 118, 'Skill#None#None#307': 119, 'Shoot#nogoal#three_point#229': 120, 'Block#fail#None#603': 121, 'Shoot#goalin#middle#217': 122, 'Shoot#goalin#layupndunk#210': 123, 'ShootCancel#goalin#three_point#215': 124, 'ShootCancel#nogoal#layupndunk#112': 125, 'Pass#None#pass_short_nicepass#704': 126, 'ShootCancel#nogoal#three_point#215': 127, 'Conversion#None#pickup#None': 128, 'Shoot#goalin#three_point#227': 129, 'Shoot#nogoal#near#325': 130, 'Shoot#goalin#three_point#229': 131, 'Pass#None#intercept#404': 132, 'Conversion#None#BuzzerBeater#None': 133, 'Rebound#success#normal#1': 134, 'Skill#None#None#702': 135, 'Skill#None#None#316': 136, 'Skill#None#None#116': 137, 'Shoot#nogoal#middle#213': 138, 'Shoot#goalin#middle#212': 139, 'ShootCancel#nogoal#middle#15': 140, 'Rebound#success#block_highball#402': 141, 'Shoot#goalin#layupndunk#101': 142, 'Rebound#fail#block_heighball#1': 143, 'Skill#None#None#23': 144, 'Pass#None#pass_long_run#701': 145, 'Block#success#None#603': 146, 'Shoot#goalin#near#4': 147, 'ShootCancel#goalin#middle#4': 148, 'Shoot#goalin#near#201': 149, 'ShootCancel#goalin#layupndunk#107': 150, 'Shoot#goalin#three_point#218': 151, 'ShootCancel#nogoal#middle#217': 152, 'Shoot#nogoal#middle#209': 153, 'Shoot#goalin#three_point#212': 154, 'ShootCancel#nogoal#near#207': 155, 'ShootCancel#goalin#near#205': 156, 'Pass#None#intercept_catch#705': 157, 'ShootCancel#goalin#middle#203': 158, 'ShootCancel#nogoal#layupndunk#15': 159, 'Shoot#nogoal#middle#211': 160, 'ShootCancel#goalin#three_point#229': 161, 'Skill#None#None#226': 162, 'Shoot#nogoal#middle#116': 163, 'Skill#None#None#605': 164, 'ShootCancel#nogoal#three_point#4': 165, 'ShootCancel#goalin#middle#224': 166, 'ShootCancel#nogoal#layupndunk#101': 167, 'Rebound#fail#block_highball#1': 168, 'ShootCancel#nogoal#middle#201': 169, 'Shoot#goalin#near#210': 170, 'Pass#None#pass_short#705': 171, 'Skill#None#None#21': 172, 'Shoot#nogoal#three_point#4': 173, 'Shoot#goalin#near#702': 174, 'Shoot#nogoal#three_point#212': 175, 'Shoot#nogoal#middle#228': 176, 'ShootCancel#nogoal#middle#4': 177, 'Shoot#nogoal#middle#224': 178, 'ShootCancel#nogoal#near#205': 179, 'ShootCancel#nogoal#middle#213': 180, 'Skill#None#None#111': 181, 'Shoot#nogoal#layupndunk#210': 182, 'Pass#None#pass_short_nicepass#701': 183, 'Rebound#fail#block_heighball#402': 184, 'Shoot#goalin#three_point#217': 185, 'ShootCancel#goalin#layupndunk#108': 186, 'ShootCancel#nogoal#three_point#212': 187, 'ShootCancel#nogoal#three_point#203': 188, 'Pass#None#pass_short#6': 189, 'ShootCancel#goalin#near#325': 190, 'Shoot#nogoal#middle#203': 191, 'Skill#None#None#223': 192, 'ShootCancel#goalin#near#29': 193, 'Shoot#goalin#near#218': 194, 'Pass#None#intercept_catch#703': 195, 'Skill#None#None#701': 196, 'Pass#None#pass_short#703': 197, 'Shoot#goalin#three_point#223': 198, 'Pass#None#intercept_catch#701': 199, 'ShootCancel#goalin#middle#702': 200, 'ShootCancel#goalin#middle#325': 201, 'ShootCancel#nogoal#layupndunk#108': 202, 'Shoot#nogoal#three_point#214': 203, 'Steal#fail#None#8': 204, 'Skill#None#None#710': 205, 'Shoot#nogoal#layupndunk#112': 206, 'Skill#None#None#40': 207, 'Block#success#None#9': 208, 'Shoot#nogoal#near#207': 209, 'Shoot#goalin#layupndunk#109': 210, 'Steal#success#None#8': 211, 'Skill#None#None#501': 212, 'Pass#None#intercept_catch#6': 213, 'Shoot#nogoal#middle#325': 214, 'ShootCancel#nogoal#layupndunk#109': 215, 'ShootCancel#goalin#three_point#116': 216, 'Shoot#goalin#middle#15': 217, 'Shoot#goalin#middle#208': 218, 'Shoot#nogoal#layupndunk#110': 219, 'Shoot#nogoal#middle#226': 220, 'Shoot#goalin#middle#228': 221, 'ShootCancel#goalin#three_point#213': 222, 'ShootCancel#goalin#near#702': 223, 'ShootCancel#goalin#near#15': 224, 'ShootCancel#nogoal#three_point#217': 225, 'Shoot#goalin#middle#4': 226, 'Shoot#nogoal#middle#702': 227, 'Shoot#nogoal#near#4': 228, 'Shoot#nogoal#near#211': 229, 'Pass#None#intercept_catch#404': 230, 'ShootCancel#goalin#middle#212': 231, 'Skill#None#None#215': 232, 'ShootCancel#nogoal#middle#209': 233, 'Skill#None#None#301': 234, 'Shoot#nogoal#near#201': 235, 'Shoot#nogoal#middle#208': 236, 'Rebound#fail#normal#402': 237, 'Shoot#goalin#middle#201': 238, 'Rebound#fail#normal#401': 239, 'Shoot#goalin#near#205': 240, 'Shoot#nogoal#three_point#203': 241, 'Shoot#nogoal#three_point#228': 242, 'Skill#None#None#110': 243, 'Shoot#goalin#middle#218': 244, 'ShootCancel#goalin#near#4': 245, 'Skill#None#None#210': 246, 'Skill#None#None#212': 247, 'Shoot#nogoal#layupndunk#103': 248, 'Skill#None#None#41': 249, 'ShootCancel#nogoal#middle#208': 250, 'Shoot#goalin#near#15': 251, 'Skill#None#None#211': 252, 'ShootCancel#goalin#middle#208': 253, 'ShootCancel#goalin#near#201': 254, 'ShootCancel#goalin#layupndunk#111': 255, 'Pass#None#pass_long_run#6': 256, 'Skill#None#None#401': 257, 'Skill#None#None#506': 258, 'ShootCancel#goalin#middle#201': 259, 'Pass#None#pass_short#404': 260, 'Shoot#nogoal#layupndunk#15': 261, 'Shoot#nogoal#layupndunk#109': 262, 'ShootCancel#nogoal#middle#226': 263, 'Skill#None#None#202': 264, 'ShootCancel#nogoal#middle#218': 265, 'Rebound#fail#block_highball#401': 266, 'ShootCancel#goalin#three_point#225': 267, 'ShootCancel#nogoal#three_point#225': 268, 'Skill#None#None#207': 269, 'Skill#None#None#101': 270, 'Pass#None#pass_short_nicepass#404': 271, 'Shoot#goalin#layupndunk#114': 272, 'Shoot#goalin#layupndunk#110': 273, 'ShootCancel#goalin#three_point#223': 274, 'Shoot#nogoal#near#15': 275, 'Shoot#goalin#three_point#222': 276, 'ShootCancel#goalin#layupndunk#109': 277, 'Shoot#goalin#middle#325': 278, 'Skill#None#None#214': 279, 'ShootCancel#goalin#layupndunk#114': 280, 'Steal#success#None#505': 281, 'Shoot#goalin#layupndunk#107': 282, 'ShootCancel#nogoal#three_point#116': 283, 'Skill#None#None#327': 284, 'Block#success#None#602': 285, 'ShootCancel#nogoal#near#4': 286, 'Skill#None#None#603': 287, 'Conversion#None#rebound#None': 288, 'Skill#None#None#114': 289, 'Shoot#nogoal#middle#217': 290, 'Skill#None#None#15': 291, 'Pass#None#pass_short_nicepass#703': 292, 'Shoot#nogoal#layupndunk#107': 293, 'Shoot#nogoal#middle#4': 294, 'Skill#None#None#208': 295, 'ShootCancel#nogoal#near#702': 296, 'Skill#None#None#201': 297, 'Skill#None#None#705': 298, 'ShootCancel#nogoal#middle#228': 299, 'Shoot#nogoal#layupndunk#114': 300, 'ShootCancel#nogoal#layupndunk#107': 301, 'ShootCancel#goalin#layupndunk#210': 302, 'Shoot#nogoal#three_point#225': 303, 'Block#success#None#605': 304, 'Skill#None#None#213': 305, 'ShootCancel#nogoal#near#201': 306, 'ShootCancel#nogoal#three_point#213': 307, 'Skill#None#None#312': 308, 'Shoot#nogoal#three_point#223': 309, 'Skill#None#None#225': 310, 'Shoot#nogoal#layupndunk#102': 311, 'ShootCancel#nogoal#middle#224': 312, 'ShootCancel#goalin#three_point#222': 313, 'ShootCancel#nogoal#near#29': 314, 'Shoot#nogoal#middle#201': 315, 'ShootCancel#nogoal#layupndunk#111': 316, 'ShootCancel#nogoal#three_point#222': 317, 'ShootCancel#nogoal#near#15': 318, 'Skill#None#None#227': 319, 'Shoot#goalin#layupndunk#111': 320, 'Skill#None#None#205': 321, 'Pass#None#pass_short_nicepass#705': 322, 'Skill#None#None#107': 323, 'ShootCancel#nogoal#layupndunk#103': 324, 'ShootCancel#nogoal#layupndunk#110': 325, 'Block#fail#None#605': 326, 'ShootCancel#nogoal#three_point#228': 327, 'ShootCancel#nogoal#three_point#223': 328, 'Pass#None#pass_long_lose#701': 329, 'ShootCancel#goalin#layupndunk#110': 330, 'Shoot#goalin#middle#203': 331, 'Block#fail#None#602': 332, 'Skill#None#None#304': 333, 'Shoot#goalin#three_point#215': 334, 'ShootCancel#nogoal#layupndunk#207': 335, 'Pass#None#pass_short_nicepass#6': 336, 'Shoot#goalin#near#29': 337}


EncodeMap2 = {'ShootCancel': 0, 'Shoot': 1, 'Block': 2, 'Conversion': 3, 'Skill': 4, 'Pass': 5, 'Rebound': 6, 'Steal': 7}

EncodeMap3 = {'Steal#success': 0, 'Steal#fail': 1, 'Shoot#goalin': 2, 'Pass#None': 3, 'Block#success': 4, 'ShootCancel#nogoal': 5, 'Shoot#nogoal': 6, 'Skill#None': 7, 'ShootCancel#goalin': 8, 'Rebound#fail': 9, 'Rebound#success': 10, 'Conversion#None': 11, 'Block#fail': 12}

EncodeMap4 = {'Conversion#None#block_catch': 0, 'Pass#None#intercept': 1, 'ShootCancel#nogoal#middle': 2, 'Block#fail#None': 3, 'Rebound#fail#block_highball': 4, 'Steal#success#None': 5, 'Pass#None#pass_short_nicepass': 6, 'ShootCancel#goalin#near': 7, 'ShootCancel#goalin#three_point': 8, 'Shoot#nogoal#middle': 9, 'Rebound#success#block_highball': 10, 'Pass#None#pass_long_lose': 11, 'Conversion#None#goalin': 12, 'Conversion#None#pickup': 13, 'Shoot#goalin#three_point': 14, 'Steal#fail#None': 15, 'Conversion#None#BuzzerBeater': 16, 'Conversion#None#steal_catch': 17, 'ShootCancel#nogoal#three_point': 18, 'Shoot#goalin#near': 19, 'Shoot#goalin#layupndunk': 20, 'Rebound#success#normal': 21, 'Pass#None#pass_short': 22, 'ShootCancel#nogoal#near': 23, 'ShootCancel#goalin#layupndunk': 24, 'ShootCancel#goalin#middle': 25, 'Shoot#nogoal#near': 26, 'Shoot#nogoal#three_point': 27, 'Shoot#goalin#middle': 28, 'Pass#None#intercept_catch': 29, 'Rebound#fail#normal': 30, 'Rebound#fail#block_heighball': 31, 'Conversion#None#rebound': 32, 'Shoot#nogoal#layupndunk': 33, 'Block#success#None': 34, 'Skill#None#None': 35, 'Pass#None#pass_long_run': 36, 'Conversion#None#clock_violation': 37, 'ShootCancel#nogoal#layupndunk': 38}




def encodefunc(x):
    encodefunc.size = len(EncodeMap)
    return EncodeMap['#'.join(x.split('#')[:4])] if '\n' not in x else EncodeMap['#'.join(x[:-1].split('#')[:4])]

def encodefunc2(x):
    encodefunc2.size = len(EncodeMap2)
    return EncodeMap2[x.split('#')[0]]

def encodefunc3(x):
    encodefunc3.size = len(EncodeMap3)
    return EncodeMap3['#'.join(x.split('#')[:2])]

def encodefunc4(x):
    encodefunc4.size = len(EncodeMap4)
    return EncodeMap4['#'.join(x.split('#')[:3])]

# EncodeMapList = [encodefunc,encodefunc2,encodefunc3,encodefunc4]
EncodeMapList = [encodefunc]
EncodeMapLen = len(EncodeMap)


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
                            # actions.add(action.split(':')[-1])
                            actions.add('#'.join(action.split(':')[-1].split('#')[:4]))
    L=list(actions)
    return str(dict(zip(L,range(len(actions)))))


if __name__ == '__main__':
    result = actionEncode(['2018-11-01'])
    print(result)




















