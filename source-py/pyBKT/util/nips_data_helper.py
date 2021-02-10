import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def convert_data(url, url2=None):
    pd.set_option('mode.chained_assignment', None)
    skill_count = 124 #hard coded to fit preprocessed data
    col_count = 8214 #hard coded to fit preprocessed data
    if os.path.exists("data/"+url):
        f = open("data/" + url, "rb")
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter = ',', header=None, names=[i for i in range(col_count)])
        
    if url2 is not None:
        if os.path.exists("data/"+url):
            f = open("data/" + url, "rb")
            df2 = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter = ',', header=None, names=[i for i in range(col_count)])
            frames = [df, df2]
            df = pd.concat(frames)

    resource=[[] for i in range(skill_count)]
    starts=[[] for i in range(skill_count)]
    lengths=[[] for i in range(skill_count)]
    data=[[] for i in range(skill_count)]
    
    #code to find max number of columns needed for a single row (8214 in this case)
    #max_len = 0
    #for i in range(len(df)//3):
    #    ind = i*3
    #    num_data = df[ind:ind+1][0].values[0]
    #    if num_data>max_len:
    #        max_len = num_data
    #    print(num_data, max_len)
    
    for i in range(len(df)//3):
        ind = i*3
        num_data = df[ind:ind+1][0].values[0]
        skill_row, correct_row = df[ind+1:ind+2], df[ind+2:ind+3]
        consecutive, prev_skill = 0, skill_row[0].values[0]

        for j in range(num_data):
            #print(df[ind+1:ind+2])
            #print(j, num_data)
            skill = int(skill_row[j].values[0])
            correct = int(correct_row[j].values[0])
            data[skill].append(correct)
            if skill == prev_skill:
                consecutive += 1
            else:
                if consecutive == 1:
                    data[prev_skill] = data[prev_skill][:-1]
                else:
                    lengths[prev_skill].append(consecutive)
                prev_skill = skill
                consecutive = 1
        
        lengths[prev_skill].append(consecutive)
        
    for i in range(skill_count):
        resource[i] = [1]*len(data[i])
        starts[i].append(1)
        for j in range(len(lengths[i])-1):
            starts[i].append(starts[i][j]+lengths[i][j])
    
    Data=[dict() for i in range(skill_count)]
    for i in range(skill_count):
        Data[i]["data"] = np.asarray([[x+1 for x in data[i]]],dtype='int32')
        Data[i]["resources"] = np.asarray(resource[i])
        Data[i]["starts"] = np.asarray(starts[i])
        Data[i]["lengths"] = np.asarray(lengths[i])
        stateseqs=np.copy(resource[i])
        Data[i]["stateseqs"] = np.asarray([stateseqs],dtype='int32')
    return Data
