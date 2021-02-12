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
                # assume comma delimiter
                df = pd.read_csv(url, low_memory=False, encoding="latin")
            except:
                # try tab delimiter if comma delimiter fails
                df = pd.read_csv(url, low_memory=False, encoding="latin", delimiter='\t')
        
        # otherwise, fetch it from web using requests
        elif url[:4] == "http":
            s = requests.get(url).content
            try:
                df = pd.read_csv(s, low_memory=False, encoding="latin")
            except:
                df = pd.read_csv(s, low_memory=False, encoding="latin", delimiter='\t')
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
                 'multipair': 'template_id',
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
    if any(x in list(df.columns) for x in as_default.values()):
        for k,v in as_default.items():
            if k not in defaults:
                defaults[k] = as_default[k]
    elif any(x in list(df.columns) for x in ct_default.values()):
        for k,v in ct_default.items():
            if k not in defaults:
                defaults[k] = ct_default[k]

    # sort by the order in which the problems were answered
    if "order_id" in defaults:
        df[defaults["order_id"]] = df[defaults["order_id"]].apply(lambda x: int(x))
        df.sort_values(defaults["order_id"], inplace=True)
    
    # make sure all responses of the same user are grouped together with stable sorting
    df.sort_values(defaults["user_id"], kind="mergesort", inplace=True)
    
    if "original" in df.columns:
        df = df[(df["original"]==1)]
    
    datas = {}
    all_skills = pd.Series(df[defaults["skill_name"]].unique()).dropna()
    all_skills = all_skills[all_skills.str.match(skill_name).astype(bool)]
    if all_skills.empty:
        raise ValueError("no matching skills")
    for skill_ in all_skills:

        # filter out based on skill
        df = df[df[defaults["skill_name"]] == skill_]

        # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
        df.loc[:,defaults["correct"]]+=1
        
        # array representing correctness of student answers
        data=np.array(df[defaults["correct"]])
        
        Data={}
        gs_ref,resource_ref = {}, {}
    
        # create starts and lengths arrays
        lengths = np.array(df.groupby(defaults["user_id"])[defaults["user_id"]].count().values, dtype=np.int64)
        starts = np.zeros(len(lengths), dtype=np.int64)
        starts[0] = 1
        for i in range(1, len(lengths)):
            starts[i] = starts[i-1] + lengths[i-1]

        # different types of resources handling: multipair, multiprior, multilearn and n/a
        if multipair:
            resources = np.ones(len(data), dtype=np.int64)
            counter = 2
            resource_ref["N/A"] = 1 #no pair
            for i in range(len(df)):
                # for the first entry of a new student, no pair
                if i == 0 or df[i:i+1][defaults["user_id"]].values != df[i-1:i][defaults["user_id"]].values:
                    resources[i] = 1
                else:
                    # each pair is keyed via "[item 1] [item 2]"
                    k = (str)(df[i:i+1][defaults["multipair"]].values)+" "+(str)(df[i-1:i][defaults["multipair"]].values)
                    if k not in resource_ref:
                        # form the resource reference as we iterate through the dataframe, mapping each new pair to a number [1, # total pairs]
                        resource_ref[k] = counter
                        counter += 1
                    resources[i] = resource_ref[k]
        elif multiprior:
            resources = np.ones(len(data)+len(starts), dtype=np.int64)
            new_data = np.zeros(len(data)+len(starts), dtype=np.int32)
            # create new resources [2, #total + 1] based on how student initially responds
            resource_ref = dict(zip(df[defaults["multiprior"]].unique(),range(2, len(df[defaults["multiprior"]].unique())+2)))
            resource_ref["N/A"] = 1
            all_resources = np.array(df[defaults["multiprior"]].apply(lambda x: resource_ref[x]))
            # create phantom timeslices with resource 2 or 3 in front of each new student based on their initial response
            for i in range(len(starts)):
                new_data[i+starts[i]:i+starts[i]+lengths[i]] = data[starts[i]-1:starts[i]+lengths[i]-1]
                resources[i+starts[i]-1] = 1
                resources[i+starts[i]:i+starts[i]+lengths[i]] = all_resources[starts[i]-1:starts[i]+lengths[i]-1]
                starts[i] += i
                lengths[i] += 1
            data = new_data
        elif multilearn:
            # map each new resource found to a number [1, # total]
            resource_ref=dict(zip(df[defaults["multilearn"]].unique(),range(1,len(df[defaults["multilearn"]].unique())+1)))
            resources = np.array(df[defaults["multilearn"]].apply(lambda x: resource_ref[x]))
        else:
            resources=np.array([1]*len(data))


        # multigs handling, make data n-dimensional where n is number of g/s types
        if multigs:
            # map each new guess/slip case to a row [0, # total]
            gs_ref=dict(zip(df[defaults["multigs"]].unique(),range(len(df[defaults["multigs"]].unique()))))
            data_ref = np.array(df[defaults["multigs"]].apply(lambda x: gs_ref[x]))
        
            # make data n-dimensional, fill in corresponding row and make other non-row entries 0
            data_temp = np.zeros((len(df[defaults["multigs"]].unique()), len(df)))
            for i in range(len(data_temp[0])):
                data_temp[data_ref[i]][i] = data[i]
            Data["data"]=np.asarray(data_temp,dtype='int32')
        else:
            data = [data]
            Data["data"]=np.asarray(data,dtype='int32')

        # for when no resource and/or guess column is selected
        if not multilearn and not multipair and not multiprior:
            resource_ref[""]=1
        if not multigs:
            gs_ref[""]=1
            
        Data["starts"]=starts
        Data["lengths"]=lengths
        Data["resources"]=resources
        Data["resource_names"]=resource_ref
        Data["gs_names"]=gs_ref

        datas[skill_] = Data

    return datas
    
 

