def assistments_data(skill_name):
  
  import pandas as pd
  import numpy as np
  import io
  import requests

  url = "http://users.wpi.edu/~yutaowang/data/skill_builder_data.csv"
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(s.decode('ISO-8859-1')))
  
  # filter by the skill you want, make sure the question is an 'original'
  skill = df[(df['skill_name']==skill_name) & (df['original'] == 1)]
  # sort by the order in which the problems were answered
  df.sort_values('order_id', inplace=True)

  # example of how to get the unique users
  # uilist=skill['user_id'].unique()

  # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
  skill.loc[:,'correct']+=1
  
  # filter out garbage
  df3=skill[skill['correct']!=3]
  data=df3['correct'].values
  
  # find out how many problems per user, form the start/length arrays
  steps=df3.groupby('user_id')['problem_id'].count().values
  lengths=np.copy(steps)
  lengths=np.resize(lengths,lengths.size-1)
  steps[0]=0
  steps[1]=1
  for i in range(1,steps.size):
    steps[i]=steps[i-1]+lengths[i-1]
  starts=np.delete(steps,0)

  resources=[1]*data.size
  resource=np.asarray(resources)
  
  stateseqs=np.copy(resource)
  Data={}
  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["data"]=np.asarray([data],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource
  
  return (Data)

#assistments_data('Pythagorean Theorem')
