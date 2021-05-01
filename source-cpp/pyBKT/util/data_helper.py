#########################################
# data_helper.py                        #
# data_helper                           #
#                                       #
# @author Frederic Wang                 #
# Last edited: 07 April 2021            #
#########################################

import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def convert_data(url, skill_name, defaults=None, model_type=None, gs_refs=None, resource_refs=None, return_df = False, folds=False):
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
        else:
            raise ValueError("File path or dataframe input not found")

    # default column names for assistments
    as_default={'order_id': 'order_id',
                 'skill_name': 'skill_name',
                 'correct': 'correct',
                 'user_id': 'user_id',
                 'multilearn': 'template_id',
                 'multiprior': 'correct',
                 'multipair': 'template_id',
                 'multigs': 'template_id',
                 'folds': 'user_id',
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
                'folds': 'Anon Student Id',
                                 }

    # integrate custom defaults with default assistments/ct columns if they are still unspecified
    if defaults is None:
        defaults = {}
    elif isinstance(defaults, dict):
        ks = tuple(defaults.items())
        for k, v in ks:
            if v not in df.columns and k in as_default:
                defaults.pop(k)
    else:
        raise ValueError("incorrectly specified defaults")

    if any(x in list(df.columns) for x in as_default.values()):
        for k,v in as_default.items():
            if k not in defaults and as_default[k] in df.columns:
                defaults[k] = as_default[k]
    if any(x in list(df.columns) for x in ct_default.values()):
        for k,v in ct_default.items():
            if k not in defaults and ct_default[k] in df.columns:
                defaults[k] = ct_default[k]

    # sort by the order in which the problems were answered
    if "order_id" in defaults:
        df[defaults["order_id"]] = df[defaults["order_id"]].apply(lambda x: int(x))
        df.sort_values(defaults["order_id"], inplace=True)
    
    if "user_id" not in defaults:
        raise KeyError("user id default column not specified")
    elif defaults["user_id"] not in df.columns:
        raise KeyError("specified user id default column not in data")
        
    if "correct" not in defaults:
        raise KeyError("correct default column not specified")
    elif defaults["correct"] not in df.columns:
        raise KeyError("specified correct default column not in data")
        
    if "skill_name" not in defaults:
        raise KeyError("skill name default column not specified")
    elif defaults["skill_name"] not in df.columns:
        raise KeyError("specified skill name default column not in data")
    
    # make sure all responses of the same user are grouped together with stable sorting
    df.sort_values(defaults["user_id"], kind="mergesort", inplace=True)
    
    if "original" in df.columns:
        df = df[(df["original"]==1)]
        
    df[defaults["skill_name"]] = df[defaults["skill_name"]].apply(str)
    try:
        df[defaults["correct"]] = df[defaults["correct"]].apply(int)
    except:
        raise ValueError("Invalid Data In Specified Corrects Column")
    
    datas = {}
    skill_name = '^(' + skill_name + ')$'
    all_skills = pd.Series(df[defaults["skill_name"]].unique()).dropna()
    all_skills = all_skills[all_skills.str.match(skill_name).astype(bool)]
    if all_skills.empty:
        raise ValueError("no matching skills")
    for skill_ in all_skills:
        
        if resource_refs is None or skill_ not in resource_refs:
            resource_ref = None
        else:
            resource_ref = resource_refs[skill_]["resource_names"]
        if gs_refs is None or skill_ not in gs_refs:
            gs_ref = None
        else:
            gs_ref = gs_refs[skill_]["gs_names"]

        # filter out based on skill
        df3 = df[df[defaults["skill_name"]] == skill_]
        if df3.empty:
            raise ValueError("Incorrect Skill or Dataset Specified")

        stored_index = df3.index.copy()
        multiprior_index = None

        # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
        if set(df3.loc[:,defaults["correct"]].unique()) - set([-1, 0, 1]) != set():
            raise ValueError("correctness must be -1 (no response), 0 (incorrect), or 1 (correct)")
        df3.loc[:,defaults["correct"]]+=1
        
        # array representing correctness of student answers
        data=np.array(df3[defaults["correct"]])
        
        Data={}
    
        # create starts and lengths arrays
        lengths = np.array(df3.groupby(defaults["user_id"])[defaults["user_id"]].count().values, dtype=np.int64)
        starts = np.zeros(len(lengths), dtype=np.int64)
        starts[0] = 1
        for i in range(1, len(lengths)):
            starts[i] = starts[i-1] + lengths[i-1]

        if multipair + multiprior + multilearn > 1:
            raise ValueError("cannot specify more than 1 resource handling")

        # different types of resources handling: multipair, multiprior, multilearn and n/a
        if multipair:
            if "multipair" not in defaults:
                raise KeyError("multipair default column not specified")
            elif defaults["multipair"] not in df3.columns:
                raise KeyError("specified multipair default column not in data")
                
            resources = np.ones(len(data), dtype=np.int64)
            if resource_ref is None:
                new_resource_ref = {}
                new_resource_ref["Default"] = 1 #no pair
            for i in range(len(df3)):
                # for the first entry of a new student, no pair
                if i == 0 or df3[i:i+1][defaults["user_id"]].values != df3[i-1:i][defaults["user_id"]].values:
                    resources[i] = 1
                else:
                    # each pair is keyed via "[item 1] [item 2]"
                    k = (str)(df3[i:i+1][defaults["multipair"]].values)+" "+(str)(df3[i-1:i][defaults["multipair"]].values)
                    if resource_ref is not None and k not in resource_ref:
                        raise ValueError("Pair", k, "not fitted")
                    if k not in new_resource_ref:
                        # form the resource reference as we iterate through the dataframe, mapping each new pair to a number [1, # total pairs]
                        new_resource_ref[k] = len(new_resource_ref)+1
                    resources[i] = new_resource_ref[k]
            if resource_ref is None:
                resource_ref = new_resource_ref
        elif multiprior:
            if "multiprior" not in defaults:
                raise KeyError("multiprior default column not specified")
            elif defaults["multiprior"] not in df3.columns:
                raise KeyError("specified multiprior default column not in data")
                
            resources = np.ones(len(data)+len(starts), dtype=np.int64)
            new_data = np.zeros(len(data)+len(starts), dtype=np.int32)
            # create new resources [2, #total + 1] based on how student initially responds
            all_priors = df3[defaults["multiprior"]].unique()
            all_priors = np.sort(all_priors)
            if resource_ref is None:
                resource_ref = {}
                resource_ref["Default"] = 1
                resource_ref.update(dict(zip(all_priors,range(2, len(df3[defaults["multiprior"]].unique())+2))))
            else:
                for i in all_priors:
                    if i not in resource_ref:
                        raise ValueError("Prior", i, "not fitted")
                        
            all_resources = np.array(df3[defaults["multiprior"]].apply(lambda x: resource_ref[x]))
            
            # create phantom timeslices with resource 2 or 3 in front of each new student based on their initial response
            for i in range(len(starts)):
                new_data[i+starts[i]:i+starts[i]+lengths[i]] = data[starts[i]-1:starts[i]+lengths[i]-1]
                resources[i+starts[i]-1] = all_resources[starts[i]-1]
                resources[i+starts[i]:i+starts[i]+lengths[i]] = np.ones(lengths[i])
                starts[i] += i
                lengths[i] += 1
            
            multiprior_index = np.array([starts[i]-1 for i in range(len(starts))])
            data = new_data
        elif multilearn:
            if "multilearn" not in defaults:
                raise KeyError("multilearn default column not specified")
            elif defaults["multilearn"] not in df3.columns:
                raise KeyError("specified multilearn default column not in data")
                
            all_learns = df3[defaults["multilearn"]].unique()
            if resource_ref is None:
                # map each new resource found to a number [1, # total]
                resource_ref=dict(zip(all_learns,range(1,len(df[defaults["multilearn"]].unique())+1)))
            else:
                for i in all_learns:
                    if i not in resource_ref:
                        raise ValueError("Learn rate", i, "not fitted")
                
            resources = np.array(df3[defaults["multilearn"]].apply(lambda x: resource_ref[x]))
        else:
            resources=np.array([1]*len(data))


        # multigs handling, make data n-dimensional where n is number of g/s types
        if multigs:
            if "multigs" not in defaults:
                raise KeyError("multigs default column not specified")
            elif defaults["multigs"] not in df3.columns:
                raise KeyError("specified multigs default column not in data")
                
            all_guess = df3[defaults["multigs"]].unique()
            # map each new guess/slip case to a row [0, # total]
            if gs_ref is None:
                gs_ref=dict(zip(all_guess,range(len(df[defaults["multigs"]].unique()))))
            else:
                for i in all_guess:
                    if i not in gs_ref:
                        raise ValueError("Guess rate", i, "not previously fitted")
            data_ref = np.array(df3[defaults["multigs"]].apply(lambda x: gs_ref[x]))
        
            # make data n-dimensional, fill in corresponding row and make other non-row entries 0
            data_temp = np.zeros((len(df3[defaults["multigs"]].unique()), len(df3)))
            for i in range(len(data_temp[0])):
                data_temp[data_ref[i]][i] = data[i]
            Data["data"]=np.asarray(data_temp,dtype='int32')
        else:
            data = [data]
            Data["data"]=np.asarray(data,dtype='int32')

        # for when no resource and/or guess column is selected
        if not multilearn and not multipair and not multiprior:
            resource_ref = {}
            resource_ref["default"]=1
        if not multigs:
            gs_ref = {}
            gs_ref["default"]=1
            
        Data["starts"]=starts
        Data["lengths"]=lengths
        Data["resources"]=resources
        Data["resource_names"]=resource_ref
        Data["gs_names"]=gs_ref
        Data["index"]=stored_index
        Data["multiprior_index"]=multiprior_index
        if folds:
            Data["folds"] = np.array(df3[defaults["folds"]])

        datas[skill_] = Data

    if return_df:
        return datas, df

    return datas
