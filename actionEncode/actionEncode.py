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

EncodeMap = {'Shoot#217#nogoal#three_point': 0, 'Skill#229': 1, 'Shoot#101#goalin#layupndunk': 2, 'ShootCancel#goalin#4#near': 3, 'Shoot#15#goalin#layupndunk': 4, 'Shoot#4#nogoal#middle': 5, 'ShootCancel#goalin#229#three_point': 6, 'ShootCancel#nogoal#203#three_point': 7, 'Pass#6#pass_long_run': 8, 'Shoot#224#nogoal#middle': 9, 'Shoot#101#nogoal#layupndunk': 10, 'Skill#324': 11, 'Shoot#228#goalin#middle': 12, 'Skill#605': 13, 'Skill#327': 14, 'Pass#703#intercept': 15, 'Skill#207': 16, 'Skill#228': 17, 'Pass#701#pass_short': 18, 'ShootCancel#nogoal#211#near': 19, 'Shoot#203#nogoal#three_point': 20, 'Skill#227': 21, 'Shoot#4#goalin#near': 22, 'ShootCancel#nogoal#217#middle': 23, 'Pass#704#intercept': 24, 'Shoot#217#goalin#middle': 25, 'Shoot#225#goalin#three_point': 26, 'ShootCancel#goalin#116#three_point': 27, 'Rebound#normal#112#fail': 28, 'Shoot#212#goalin#three_point': 29, 'ShootCancel#goalin#208#middle': 30, 'Shoot#212#nogoal#three_point': 31, 'Conversion#BuzzerBeater': 32, 'ShootCancel#nogoal#210#layupndunk': 33, 'Shoot#213#goalin#three_point': 34, 'Conversion#rebound': 35, 'Shoot#15#nogoal#near': 36, 'Pass#710#pass_short_nicepass': 37, 'ShootCancel#nogoal#227#three_point': 38, 'Skill#329': 39, 'Skill#501': 40, 'ShootCancel#nogoal#201#near': 41, 'ShootCancel#goalin#215#three_point': 42, 'Skill#209': 43, 'Skill#218': 44, 'Shoot#203#goalin#middle': 45, 'Rebound#normal#401#fail': 46, 'Skill#222': 47, 'Skill#225': 48, 'Shoot#110#goalin#layupndunk': 49, 'Shoot#210#nogoal#near': 50, 'ShootCancel#nogoal#210#near': 51, 'Steal#505#success': 52, 'Pass#701#pass_long_run': 53, 'ShootCancel#goalin#218#middle': 54, 'Shoot#325#nogoal#middle': 55, 'ShootCancel#nogoal#109#layupndunk': 56, 'ShootCancel#nogoal#213#middle': 57, 'Skill#212': 58, 'Pass#701#pass_short_nicepass': 59, 'Rebound#normal#402#fail': 60, 'ShootCancel#goalin#228#middle': 61, 'Shoot#228#nogoal#three_point': 62, 'ShootCancel#nogoal#325#near': 63, 'ShootCancel#nogoal#202#layupndunk': 64, 'Shoot#15#goalin#near': 65, 'Block#602#fail': 66, 'Shoot#29#nogoal#middle': 67, 'ShootCancel#nogoal#325#middle': 68, 'Shoot#215#nogoal#three_point': 69, 'Shoot#208#nogoal#middle': 70, 'ShootCancel#nogoal#226#middle': 71, 'ShootCancel#goalin#213#three_point': 72, 'Rebound#normal#1#fail': 73, 'Shoot#4#goalin#three_point': 74, 'Shoot#212#nogoal#middle': 75, 'Conversion#steal_catch': 76, 'Shoot#116#nogoal#middle': 77, 'ShootCancel#goalin#225#three_point': 78, 'Shoot#110#nogoal#layupndunk': 79, 'Skill#215': 80, 'Skill#210': 81, 'Shoot#224#goalin#middle': 82, 'Pass#6#pass_short_nicepass': 83, 'Skill#226': 84, 'Steal#8#fail': 85, 'Skill#110': 86, 'Skill#307': 87, 'Shoot#107#nogoal#layupndunk': 88, 'Shoot#229#nogoal#three_point': 89, 'Pass#705#intercept_catch': 90, 'Shoot#207#nogoal#near': 91, 'ShootCancel#nogoal#107#layupndunk': 92, 'Shoot#211#nogoal#middle': 93, 'Shoot#29#goalin#near': 94, 'Shoot#205#nogoal#middle': 95, 'ShootCancel#goalin#217#middle': 96, 'Skill#602': 97, 'Shoot#218#nogoal#three_point': 98, 'ShootCancel#nogoal#225#three_point': 99, 'ShootCancel#nogoal#111#layupndunk': 100, 'ShootCancel#goalin#101#layupndunk': 101, 'Shoot#214#nogoal#three_point': 102, 'Pass#404#pass_short_nicepass': 103, 'Shoot#201#nogoal#near': 104, 'Skill#41': 105, 'ShootCancel#nogoal#215#three_point': 106, 'ShootCancel#nogoal#223#three_point': 107, 'Skill#301': 108, 'Skill#304': 109, 'ShootCancel#goalin#702#middle': 110, 'Rebound#block_heighball#112#fail': 111, 'ShootCancel#nogoal#103#layupndunk': 112, 'Steal#8#success': 113, 'Skill#111': 114, 'Shoot#208#nogoal#near': 115, 'Shoot#201#goalin#near': 116, 'Shoot#222#nogoal#three_point': 117, 'Shoot#116#nogoal#three_point': 118, 'Skill#704': 119, 'Steal#501#fail': 120, 'Skill#308': 121, 'Block#9#fail': 122, 'Skill#40': 123, 'Skill#506': 124, 'Shoot#223#goalin#three_point': 125, 'ShootCancel#nogoal#202#near': 126, 'Shoot#205#nogoal#near': 127, 'Skill#703': 128, 'Block#603#fail': 129, 'ShootCancel#goalin#4#three_point': 130, 'Rebound#block_highball#401#fail': 131, 'ShootCancel#goalin#212#middle': 132, 'Shoot#702#nogoal#middle': 133, 'Shoot#217#goalin#three_point': 134, 'Skill#114': 135, 'Shoot#211#nogoal#near': 136, 'Rebound#block_highball#402#fail': 137, 'Pass#705#pass_short_nicepass': 138, 'ShootCancel#goalin#224#middle': 139, 'Skill#401': 140, 'Shoot#116#goalin#three_point': 141, 'Rebound#normal#1#success': 142, 'ShootCancel#nogoal#207#layupndunk': 143, 'Skill#211': 144, 'Skill#109': 145, 'Shoot#111#nogoal#layupndunk': 146, 'Rebound#block_heighball#1#fail': 147, 'Shoot#213#goalin#middle': 148, 'Pass#701#pass_long_lose': 149, 'ShootCancel#goalin#325#middle': 150, 'Pass#6#pass_short': 151, 'Shoot#210#goalin#layupndunk': 152, 'ShootCancel#goalin#29#near': 153, 'ShootCancel#goalin#4#middle': 154, 'Skill#201': 155, 'ShootCancel#nogoal#112#layupndunk': 156, 'ShootCancel#nogoal#15#layupndunk': 157, 'Pass#404#intercept_catch': 158, 'Conversion#pickup': 159, 'Shoot#15#nogoal#middle': 160, 'Pass#703#pass_short': 161, 'Skill#15': 162, 'Shoot#218#goalin#middle': 163, 'Shoot#208#goalin#near': 164, 'Shoot#208#goalin#middle': 165, 'Skill#205': 166, 'ShootCancel#goalin#210#layupndunk': 167, 'ShootCancel#nogoal#110#layupndunk': 168, 'Pass#703#intercept_catch': 169, 'Skill#213': 170, 'ShootCancel#nogoal#212#middle': 171, 'ShootCancel#nogoal#207#near': 172, 'Shoot#702#goalin#near': 173, 'Shoot#218#nogoal#middle': 174, 'Skill#4': 175, 'Skill#202': 176, 'ShootCancel#nogoal#4#near': 177, 'Shoot#218#goalin#three_point': 178, 'Shoot#223#nogoal#three_point': 179, 'Shoot#211#nogoal#three_point': 180, 'Skill#108': 181, 'Shoot#109#goalin#layupndunk': 182, 'Skill#21': 183, 'ShootCancel#goalin#15#near': 184, 'Skill#116': 185, 'Shoot#702#nogoal#near': 186, 'Shoot#107#goalin#layupndunk': 187, 'ShootCancel#goalin#109#layupndunk': 188, 'Shoot#325#goalin#near': 189, 'Skill#710': 190, 'Shoot#205#goalin#near': 191, 'ShootCancel#goalin#702#near': 192, 'Block#9#success': 193, 'ShootCancel#nogoal#102#layupndunk': 194, 'Skill#34': 195, 'ShootCancel#nogoal#228#three_point': 196, 'ShootCancel#nogoal#203#middle': 197, 'Shoot#29#nogoal#near': 198, 'Shoot#213#nogoal#three_point': 199, 'ShootCancel#nogoal#702#middle': 200, 'Skill#330': 201, 'ShootCancel#goalin#209#middle': 202, 'ShootCancel#nogoal#108#layupndunk': 203, 'Shoot#215#goalin#three_point': 204, 'Shoot#210#goalin#near': 205, 'ShootCancel#nogoal#101#layupndunk': 206, 'ShootCancel#goalin#325#near': 207, 'ShootCancel#nogoal#208#near': 208, 'ShootCancel#goalin#107#layupndunk': 209, 'Shoot#218#goalin#near': 210, 'Shoot#112#nogoal#layupndunk': 211, 'Conversion#clock_violation': 212, 'ShootCancel#nogoal#4#middle': 213, 'ShootCancel#nogoal#218#three_point': 214, 'Rebound#block_heighball#402#fail': 215, 'Rebound#block_highball#1#fail': 216, 'Shoot#228#nogoal#middle': 217, 'Skill#223': 218, 'Skill#701': 219, 'Shoot#202#nogoal#near': 220, 'Shoot#102#nogoal#layupndunk': 221, 'Conversion#block_catch': 222, 'Shoot#114#nogoal#layupndunk': 223, 'Skill#312': 224, 'Pass#701#intercept_catch': 225, 'ShootCancel#goalin#114#layupndunk': 226, 'ShootCancel#nogoal#229#three_point': 227, 'Rebound#block_highball#402#success': 228, 'Rebound#block_highball#1#success': 229, 'ShootCancel#nogoal#211#middle': 230, 'Shoot#226#nogoal#middle': 231, 'ShootCancel#nogoal#213#three_point': 232, 'ShootCancel#goalin#227#three_point': 233, 'Shoot#203#goalin#three_point': 234, 'Shoot#203#nogoal#middle': 235, 'Block#603#success': 236, 'ShootCancel#nogoal#211#three_point': 237, 'Shoot#4#nogoal#three_point': 238, 'ShootCancel#nogoal#208#middle': 239, 'ShootCancel#nogoal#29#near': 240, 'Shoot#108#goalin#layupndunk': 241, 'ShootCancel#nogoal#217#three_point': 242, 'ShootCancel#goalin#203#middle': 243, 'Shoot#29#goalin#middle': 244, 'Pass#404#pass_short': 245, 'Shoot#225#nogoal#three_point': 246, 'Skill#328': 247, 'ShootCancel#goalin#108#layupndunk': 248, 'ShootCancel#goalin#110#layupndunk': 249, 'Skill#23': 250, 'Pass#704#pass_short': 251, 'Shoot#209#nogoal#middle': 252, 'Shoot#702#goalin#middle': 253, 'Shoot#108#nogoal#layupndunk': 254, 'Pass#404#intercept': 255, 'Rebound#normal#402#success': 256, 'ShootCancel#goalin#111#layupndunk': 257, 'Skill#505': 258, 'Pass#6#pass_long_lose': 259, 'Pass#704#pass_short_nicepass': 260, 'Block#605#success': 261, 'Shoot#325#goalin#middle': 262, 'ShootCancel#nogoal#224#middle': 263, 'ShootCancel#goalin#201#near': 264, 'Shoot#202#nogoal#layupndunk': 265, 'ShootCancel#nogoal#205#middle': 266, 'Skill#203': 267, 'Block#602#success': 268, 'Shoot#4#goalin#middle': 269, 'ShootCancel#nogoal#15#middle': 270, 'Shoot#4#nogoal#near': 271, 'ShootCancel#nogoal#228#middle': 272, 'Skill#101': 273, 'ShootCancel#goalin#213#middle': 274, 'Skill#603': 275, 'ShootCancel#nogoal#212#three_point': 276, 'Shoot#111#goalin#layupndunk': 277, 'Pass#701#intercept': 278, 'ShootCancel#nogoal#4#three_point': 279, 'Shoot#114#goalin#layupndunk': 280, 'ShootCancel#nogoal#15#near': 281, 'Shoot#213#nogoal#middle': 282, 'ShootCancel#nogoal#218#middle': 283, 'Shoot#227#goalin#three_point': 284, 'ShootCancel#goalin#205#near': 285, 'Skill#402': 286, 'Shoot#209#goalin#middle': 287, 'Pass#704#intercept_catch': 288, 'Skill#217': 289, 'ShootCancel#nogoal#222#three_point': 290, 'Pass#705#pass_short': 291, 'Skill#702': 292, 'Skill#404': 293, 'ShootCancel#nogoal#209#middle': 294, 'Skill#107': 295, 'ShootCancel#nogoal#214#three_point': 296, 'Shoot#222#goalin#three_point': 297, 'Steal#501#success': 298, 'ShootCancel#nogoal#205#near': 299, 'Skill#103': 300, 'Rebound#normal#112#success': 301, 'ShootCancel#goalin#212#three_point': 302, 'ShootCancel#goalin#15#layupndunk': 303, 'Shoot#15#goalin#middle': 304, 'Shoot#15#nogoal#layupndunk': 305, 'ShootCancel#nogoal#114#layupndunk': 306, 'Skill#208': 307, 'Skill#316': 308, 'Shoot#103#nogoal#layupndunk': 309, 'Skill#112': 310, 'Shoot#207#nogoal#layupndunk': 311, 'ShootCancel#nogoal#218#near': 312, 'Conversion#goalin': 313, 'ShootCancel#goalin#222#three_point': 314, 'Skill#214': 315, 'Steal#505#fail': 316, 'Skill#224': 317, 'Shoot#218#nogoal#near': 318, 'Shoot#210#nogoal#layupndunk': 319, 'ShootCancel#goalin#223#three_point': 320, 'Shoot#227#nogoal#three_point': 321, 'Pass#703#pass_short_nicepass': 322, 'Pass#6#intercept': 323, 'Shoot#217#nogoal#middle': 324, 'Pass#6#intercept_catch': 325, 'Shoot#325#nogoal#near': 326, 'ShootCancel#nogoal#201#middle': 327, 'ShootCancel#nogoal#116#three_point': 328, 'ShootCancel#nogoal#702#near': 329, 'Shoot#109#nogoal#layupndunk': 330, 'Shoot#229#goalin#three_point': 331, 'Shoot#201#goalin#middle': 332, 'Skill#705': 333, 'ShootCancel#goalin#201#middle': 334, 'Block#605#fail': 335, 'Shoot#201#nogoal#middle': 336, 'Shoot#212#goalin#middle': 337}
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




















