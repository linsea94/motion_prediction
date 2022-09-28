import pandas as pd
import os 
home = os.path.expanduser("~")

origin_data = home + '/motion_ws/src/motion_prediction/data/origin_data/ETH.csv'
save_folder = home +'/motion_ws/src/motion_prediction/data/ETH/'
if not os.path.exists(save_folder):
  os.makedirs(save_folder)
  os.makedirs(save_folder+'/SplitByTime')
  os.makedirs(save_folder+'/train_ag')
  os.makedirs(save_folder+'/train_x')
  os.makedirs(save_folder+'/train_y')  
split_len =  6
fps = 0.4
first_time = 0

sample = []
sample_cnt, file_name = 0, 0
f_break = False
df = pd.read_csv(origin_data)

for i in range(len(df)): # for loop to make sure the timestamp change
  check_time = df['timestamp'][i]
  sample = []
  sample_cnt = 0
  if check_time == first_time:
    continue
  else:
    t = check_time
    first_time = check_time


  for j in range(i,len(df)): # for loop for the data for a single file
    lst = df.iloc[j]
    new_time = lst['timestamp'] 

    # next time
    if new_time != t:
      while (new_time - t) > fps * 1.5:
        f_break = True
        break
      
      sample_cnt += 1
      t = new_time
      #reset
      while sample_cnt == split_len: 
        f_break  = True
        file_name += 1
        data = sample
        df_split = pd.DataFrame(data)
        df_split.columns = ['frame_id', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'scene_id', 'label', 'timestamp']
        f = save_folder+'/SplitByTime/' + '{0:04}'.format(file_name)+'.csv'
        df_split.to_csv(f, header = True, index = False)
        break

      if f_break:
        f_break = False
        # print('break')
        break
      else:
        sample.append(lst)
    
    else:
      sample.append(lst)