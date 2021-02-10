# script to parse data in the form used in Pardos, Zachary & Heffernan, Neil. (2010). T.: Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing.
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def convert_data(url):
    pd.set_option('mode.chained_assignment', None)

    if os.path.exists("data/glops-exact/"+url):
        f = open("data/glops-exact/" + url, "rb")
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter = ' ', header=None)
        
    sequence_length = len(df.columns) - 1
    num_students = len(df)

    Data = {} # for basic bkt
    
    starts = np.linspace(1, sequence_length*(num_students-1) + 1, num=num_students, dtype=np.int)
    lengths = np.full(num_students, sequence_length - 1)
    lengths_ = np.full(num_students, sequence_length)
    resources = np.full(num_students * sequence_length, 1)

    data = []
    for i in range(num_students):
        curr_row = df[i:i+1]
        for j in range(1, sequence_length + 1):
            data.append(curr_row[j].values[0] + 1)

    Data["data"] = np.asarray([data],dtype='int32')
    Data["resources"] = resources
    Data["starts"] = starts
    Data["lengths"] = lengths
    Data["lengths_full"] = lengths_
    
    Data1 = {} # for kt_pps model
    
    sequence_length += 1
    starts1 = np.linspace(1, sequence_length*(num_students-1) + 1, num=num_students, dtype=np.int)
    lengths1 = np.full(num_students, sequence_length - 1)
    lengths1_ = np.full(num_students, sequence_length)
    resources1 = np.full(num_students * sequence_length, 1)

    data1 = []
    for i in range(num_students):
        curr_row = df[i:i+1]
        data1.append(0)
        
        if curr_row[1].values[0] == 0:
            resources1[i * sequence_length] = 2
        elif curr_row[1].values[0] == 1:
            resources1[i * sequence_length] = 3
        
        for j in range(1, sequence_length):
            data1.append(curr_row[j].values[0] + 1)
            
    Data1["data"] = np.asarray([data1],dtype='int32')
    Data1["resources"] = resources1
    Data1["starts"] = starts1
    Data1["lengths"] = lengths1
    Data1["lengths_full"] = lengths1_
    
    return Data, Data1

