import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def convert_data(url, skill_name, defaults=None, model_type=None):
    if model_type:
        multilearn, multiprior, multipair, multigs = model_type
    else:
        multilearn, multiprior, multipair, multigs = [False] * 4
    pd.set_option('mode.chained_assignment', None)
    df = None

    if not isinstance(skill_name, str):
        skill_name = '|'.join(skill_name)
    
    if isinstance(url, pd.DataFrame):
        df = url
    else:
        # if url is a local file, read it from there
        if os.path.exists(url):
            try:
                f = open(url, "rb")      
                # assume comma delimiter
                df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False)
            except:
                # try tab delimiter if comma delimiter fails
                df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter='\t')
        # otherwise, fetch it from web using requests
        elif url[:4] == "http":
            s = requests.get(url).content
            try:
                df = pd.read_csv(io.StringIO(s.decode('latin')), low_memory=False)
            except:
                df = pd.read_csv(io.StringIO(s.decode('latin')), low_memory=False, delimiter='\t')
            f = open(url.split('/')[-1], 'w+')
            # save csv to local file for quick lookup in the future
            df.to_csv(f)

    # default column names for assistments
    as_default={'order_id': 'order_id',
                 'skill_name': 'skill_name',
                 'correct': 'correct',
                 'user_id': 'user_id',
                 'multilearn': 'template_id',
                 'multiprior': 'correct',
                 'multipair': 'problem_id',
                 'multigs': 'template_id',
                 }

    # default column names for cognitive tutors
    ct_default={'order_id': 'Row',
                'skill_name': 'KC(Default)',
                'correct': 'Correct First Attempt',
                'user_id': 'Anon Student Id',
                'multilearn': 'Problem Name',
                'multiprior': 'Correct First Attempt',
                'multipair': 'Problem Name',
                'multigs': 'Problem Name',
                                 }

    # integrate custom defaults with default assistments/ct columns if they are still unspecified
    if defaults is None:
        defaults = {}
    if all(x in list(df.columns) for x in as_default.values()):
        for k,v in as_default.items():
            if k not in defaults:
                defaults[k] = as_default[k]
    elif all(x in list(df.columns) for x in ct_default.values()):
        for k,v in ct_default.items():
            if k not in defaults:
                defaults[k] = ct_default[k]

    # sort by the order in which the problems were answered
    df[defaults["order_id"]] = [int(i) for i in df[defaults["order_id"]]]
    df.sort_values(defaults["order_id"], inplace=True)
    
    if "original" in df.columns:
        df = df[(df["original"]==1)]
    
    datas = {}
    all_skills = pd.Series(df[defaults["skill_name"]].unique()).dropna()
    all_skills = all_skills[all_skills.str.match(skill_name).astype(bool)]
    if all_skills.empty:
        raise ValueError("no matching skills")
    for skill_ in all_skills:

        # filter out based on skill
        skill = df[df[defaults["skill_name"]] == skill_]
    

        # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
        skill.loc[:,defaults["correct"]]+=1
        
        # filter out garbage
        df3=skill[skill[defaults["correct"]]!=3]
        
        # array representing correctness of student answers
        data=df3[defaults["correct"]].tolist()
        
        starts, lengths, resources=[],[],[]
        counter, lcounter = 1, 0
        prev_id = -1
        Data={}
        gs_ref,resource_ref = {}, {}
        
        # form the starts/lengths arrays
        for _, i in df3.iterrows():
                if i[defaults["user_id"]] != prev_id:
                        starts.append(counter)
                        prev_id = i[defaults["user_id"]]
                        lengths.append(lcounter)
                        lcounter = 0
                lcounter += 1
                counter += 1
        lengths.append(lcounter)
        lengths = np.asarray(lengths[1:])

        # different types of resource handling: multipair, multiprior, multilearn and none
        if multipair:
            counter = 2
            nopair = 1
            resource_ref["N/A"] = nopair
            for i in range(len(df3)):
                # for the first entry of a new student, no pair
                if i == 0 or df3[i:i+1][defaults["user_id"]].values != df3[i-1:i][defaults["user_id"]].values:
                    resources.append(nopair)
                else:
                    # each pair is keyed via "[item 1] [item 2]" 
                    k = (str)(df3[i:i+1][defaults["multipair"]].values)+" "+(str)(df3[i-1:i][defaults["multipair"]].values)
                    if k in resource_ref:
                        resources.append(resource_ref[k])
                    # form the resource reference as we iterate through the dataframe, mapping each new pair to a number [1, # total pairs]
                    else:
                        resource_ref[k] = counter
                        resources.append(resource_ref[k])
                        counter += 1
        elif multiprior:
            resources=[1]*len(data)
            # create new resources [2, #total + 1] based on how student initially responds
            resource_ref = dict(zip(df3[defaults["multiprior"]].unique(),range(2, len(df3[defaults["multiprior"]].unique())+2)))
            resource_ref["N/A"] = 1
            # create phantom timeslices with resource 2 or 3 in front of each new student based on their initial response
            for i in range(len(starts)):
                starts[i] += i
                data.insert(starts[i]-1, 0)
                resources.insert(starts[i]-1, resource_ref[df3[i:i+1][defaults["multiprior"]].values[0]])
                lengths[i] += 1
        elif multilearn:
            # map each new resource found to a number [1, # total]
            resource_ref=dict(zip(df3[defaults["multilearn"]].unique(),range(1,len(df3[defaults["multilearn"]].unique())+1)))
            for _, i in df3.iterrows():
                resources.append(resource_ref[i[defaults["multilearn"]]])
        else:
            resources=[1]*len(data)

        # multigs handling, make data n-dimensional where n is number of g/s types
        if multigs:
            # map each new guess/slip case to a row [0, # total]
            gs_ref=dict(zip(df3[defaults["multigs"]].unique(),range(len(df3[defaults["multigs"]].unique()))))
            data_temp = [[] for _ in range(len(gs_ref))]
            counter = 0
            # make data n-dimensional, fill in corresponding row and make other non-row entries 0
            for _, i in df3.iterrows():
                for j in range(len(gs_ref)):
                    if gs_ref[i[defaults["multigs"]]] == j:
                            data_temp[j].append(data[counter])
                            counter += 1 
                    else:
                            data_temp[j].append(0)
            Data["data"]=np.asarray(data_temp,dtype='int32')
        else:
            data = [data]
            Data["data"]=np.asarray(data,dtype='int32')

        # for when no resource and/or guess column is selected
        if not multilearn and not multipair and not multiprior:
            resource_ref[""]=1  
        if not multigs:
            gs_ref[""]=1

        resource=np.asarray(resources)
        stateseqs=np.copy(resource)
        Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
        Data["starts"]=np.asarray(starts)
        Data["lengths"]=np.asarray(lengths)
        Data["resources"]=resource
        Data["resource_names"]=resource_ref
        Data["gs_names"]=gs_ref

        datas[skill_] = Data

    return datas
    
 

