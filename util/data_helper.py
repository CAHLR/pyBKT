def assistments_data(skill_name):
  
  import pandas as pd
  import numpy as np
  import io
  import requests

  url = "http://users.wpi.edu/~yutaowang/data/skill_builder_data.csv"
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(s.decode('ISO-8859-1')))
  #print(df.describe())
  
  df2=df[df['skill_name']==skill_name]
  df2.sort_values('user_id')
  uilist=df2['user_id'].unique()
  df2['first_action'].unique()
  df2.loc[:,'first_action']+=1
  
  df3=df2[df2['first_action']!=3]
  data=df3['first_action'].values
  
  steps=df3.groupby('user_id')['problem_id'].count().values
  #print(steps)
  
  #print(steps.type)
  lengths=np.copy(steps)
  lengths=np.resize(lengths,lengths.size-1)
  #p[0]=-1
  #temp=[-1]*steps.size
  
  
  #print(lengths)
  #print(lengths.size)
  #steps=[0]*lengths.size
  steps[0]=0
  steps[1]=1
  
  for i in range(1,steps.size):
    steps[i]=steps[i-1]+lengths[i-1]
    
  starts=np.delete(steps,0)
  
  
    
    
  #np.delete(steps,0)
 
  
  #print(starts)
  #print(starts.type)
  resources=[]
  resources=[1]*data.size
  resource=np.asarray(resources)
  
  stateseqs=np.copy(resource)
  Data={}
  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["data"]=np.asarray([data],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource
  
  #print(Data)
  #
  return (Data)

#assistments_data('Pythagorean Theorem')
