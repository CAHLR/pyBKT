def assistments_data(skill_name):
  
  import pandas as pd
  import numpy as np
  import io
  import requests
  
  print(skill_name)
  url = "https://drive.google.com/uc?export=download&id=0B3f_gAH-MpBmUmNJQ3RycGpJM0k"
  s = requests.get(url).content
  # df = pd.read_csv(io.BytesIO(s))
  df = pd.read_csv(io.StringIO(s.decode('latin')))
  # df = pd.read_csv(io.StringIO(s.decode('latin')), names = ['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 'original', 'correct', 'attempt_count', 'ms_first_response', 'tutor_mode', 'answer_type', 'sequence_id', 'student_class_id', 'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name', 'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time', 'template_id', 'answer_id', 'answer_text', 'first_action', 'bottom_hint', 'opportunity', 'opportunity_original'])
  # filter by the skill you want, make sure the question is an 'original'
  # skills = df["skill_name"]
  # for skill_name in skills:
  skill = df[(df["skill_name"]==skill_name) & (df["original"] == 1)]
  # sort by the order in which the problems were answered
  df["order_id"] = [int(i) for i in df["order_id"]]
  df.sort_values("order_id", inplace=True)
  
  # example of how to get the unique users
  # uilist=skill['user_id'].unique()

  # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
  skill.loc[:,"correct"]+=1
  
  # filter out garbage
  df3=skill[skill["correct"]!=3]
  data=df3["correct"].values
  
  # find out how many problems per user, form the start/length arrays
  steps=df3.groupby("user_id")["problem_id"].count().values
  lengths=np.copy(steps)
  # print("steps", steps)
  steps[0]=0
  steps[1]=1
  for i in range(2,steps.size):
    steps[i]=steps[i-1]+lengths[i-2]
  
  starts=np.delete(steps,0)

  resources=[1]*data.size
  resource=np.asarray(resources)
  
  stateseqs=np.copy(resource)
  lengths=np.resize(lengths,lengths.size-1)
  Data={}
  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["data"]=np.asarray([data],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource
  
  return (Data)

