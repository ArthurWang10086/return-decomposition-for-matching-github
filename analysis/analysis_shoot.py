import pandas as pd
from pandasql import  sqldf

if __name__ == '__main__':
    with open('../dataset/ball/2018-11-01.txt','r') as f:
        datas = f.read().split('\n')
        for data in datas[:1000]:
            win_team_shoot = lose_team_shoot = 0
            for x in data.split('@')[1].split(','):
                playerid = x.split(':')[0]
                assert playerid in ['0','1','2','3','4','5']
                actions = x.split(':')[1]
                if 'Conversion#None#goalin#None' in actions and playerid in ['0','1','2']:
                    win_team_shoot += 1
                elif 'Conversion#None#goalin#None' in actions and playerid in ['3','4','5']:
                    lose_team_shoot += 1
            if lose_team_shoot< win_team_shoot:
                print(win_team_shoot,lose_team_shoot)
                print(data)



