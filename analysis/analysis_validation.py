import pandas as pd
from pandasql import  sqldf


def func1():
    L=[]
    with open('../dataset/process_data/2018-11-01.txt','r') as f:
        datas = f.read().split('\n')
        for data in datas:
            tmp = 0
            for i,x in enumerate(data.split(';')):
                result = x.split('@')[1]
                result = '1' if int(result)>0 else '-1'
                for action in x.split('@')[2].split(','):
                    #Conversion#None#goalin#None
                    if len(action.split(':')[-1])>0 and 'Conversion#None#goalin#None' in action.split(':')[-1]:
                        tmp += 1
            L.append(data.split('@')[0].split('|')[0]+','+str(tmp))
    with open('2018-11-01-validation.txt','w') as f:
        f.write('\n'.join(L))

def func2():
    data_true = open('2018-11-01-true.txt').read().split('\n')[:-1]
    data_check = open('2018-11-01-validation.txt').read().split('\n')[:-1]
    data_true_dict = dict([(x.split(',')[0],x.split(',')[1]) for x in data_true])
    data_check_dict = dict([(x.split(',')[0],x.split(',')[1]) for x in data_check])
    count=0
    for key in data_check_dict:
        if key not in data_true_dict:
            pass
        elif data_check_dict[key] != data_true_dict[key]:
            print(key,data_true_dict[key],data_check_dict[key])
            count+=1
    print(len(data_true),len(data_check),count)


if __name__ == '__main__':
    func1()
    func2()





