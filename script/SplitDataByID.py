import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import AngularGrid
from SplitData import *
from glob import glob

home = os.path.expanduser("~")
pkg_location = home + "/motion_ws/src/OpenTraj/opentraj/toolkit/my_test/output/ETH/"
folder = pkg_location + "SplitByTime/"
h = (np.loadtxt("/home/linsea/motion_ws/src/motion_prediction/data/ETH/H.txt"))
img = cv2.imread("/home/linsea/motion_ws/src/motion_prediction/data/ETH/map.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
output_frame_num = 5
num_of_pieces = 72
max_dist = 8
cnt = 0

for file in glob(os.path.join(folder,"*.csv")):
  file_name = file.split('/')[-1]
  file_name = file_name.split('.csv')[0]
  df = pd.read_csv(file)
  cols = ['frame_id', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'scene_id', 'label', 'timestamp']
  target_set = get_target_set(df)
  # print('file', file_name,'target:',target_set,'working!')

  for target in target_set:
    target_state, target_output, other_state = split_data(df, target)

    target_state = pd.DataFrame(target_state)
    target_state.columns = cols

    target_center_pt = np.array([target_state['pos_x'], target_state['pos_y']]).reshape(1,2)
    map_feature = get_map_feature_ETH(target_center_pt, img, h)                                #only for ETH dataset!
    map_feature = pd.DataFrame(map_feature)
    
    try:
      target_output = pd.DataFrame(target_output)
      target_output.columns = cols
      if len(target_output) != output_frame_num:
        print("file", file_name, "target:", target, "doesn't have the complete trajactory!")
        continue
    except:
      print("file", file_name, "target:", target, "doesn't have the complete trajactory!")
      continue

    try:
      other_state = pd.DataFrame(other_state)
      other_state.columns = cols
      target_pt = AngularGrid.get_pt(target_state.iloc[0])
      ag = [AngularGrid.get_AngularGrid(other_state, target_pt, num_of_pieces= num_of_pieces, max_dist= max_dist)]
      ag = pd.DataFrame(ag)
    except:
      ag = [np.ones(num_of_pieces)]
      ag = pd.DataFrame(ag)
      print("file", file_name, "target:", target, "doesn't have the others!")

    target_state.to_csv(pkg_location + 'train_x/' + str(file_name) + '_' + str(target) +'.csv', header = True, index = False)
    map_feature.to_csv(pkg_location + 'map_info/' + str(file_name) + '_' + str(target) +'.csv', header = False, index = False)
    target_output.to_csv(pkg_location + 'train_y/' + str(file_name) + '_' + str(target) +'.csv', header = True, index = False)
    ag.to_csv(pkg_location + 'train_ag/' + str(file_name) + '_' + str(target) +'.csv', header = True, index = False)
    cnt+=1

print(cnt)