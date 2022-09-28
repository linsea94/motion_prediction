import numpy as np
import pandas as pd

def get_ped_info(ped_info):

  ped_id = ped_info[1]
  ped_frame = ped_info[0]
  
  return ped_frame, ped_id

def get_target_set(df):
  first_frame = df.iloc[0]['frame_id']
  target_set = []

  for i in range(len(df)):
    ped_info = df.iloc[i]
    ped_frame, ped_id = get_ped_info(ped_info)
    if ped_frame != first_frame:
      break
    else:
      target_set.append(ped_id)

  return target_set

def split_data(df, target):
    FIRST = True
    target_state=[]
    target_output=[]
    other_state=[]
    first_frame = df.iloc[0]['frame_id']

    for i in range(len(df)):
      ped_info = df.iloc[i]
      ped_frame, ped_id = get_ped_info(ped_info)
      if ped_id == target:
        if FIRST:
          FIRST = False
          target_state.append(ped_info)
        else:
          target_output.append(ped_info)

      elif ped_frame == first_frame:
        other_state.append(ped_info)
      else:
        pass
    return target_state, target_output, other_state

def get_map_feature(center_pt, map, cut_size=30):
  x = center_pt[0]
  y = center_pt[1]
  helf_size = int(cut_size/2)
  map_feature = []
  for i in range(x - helf_size, x + helf_size):
    for j in range(y - helf_size, y + helf_size):
      map_feature.append(map[i][j])
  map_feature = np.array(map_feature).reshape(cut_size, cut_size)
  return np.array(map_feature)

def get_map_feature_ETH(center_pt, map, h, cut_size=20):
  pt = traj2pixal(center_pt, h)
  x = pt[0][1]
  y = pt[0][0]
  helf_size = int(cut_size/2)
  map_feature = []
  size = map.shape
  for i in range(x - helf_size, x + helf_size):
    for j in range(y - helf_size, y + helf_size):
      if i >= size[0]:
        map_feature.append(0)
      elif j >= size[1]:
        map_feature.append(0)
      else:
        map_feature.append(map[i][j])
  map_feature = np.array(map_feature).reshape(cut_size, cut_size)
  return map_feature

def traj2pixal(traj,h):
  # h = (np.loadtxt(os.path.join(path_H)))
  h_inv = np.linalg.inv(h)
  traj_homog = np.hstack((traj, np.ones((traj.shape[0], 1)))).T
  traj_cam = np.matmul(h_inv, traj_homog)  
  traj_uvz = np.transpose(traj_cam/traj_cam[2])
  return traj_uvz[:, :2].astype(int)    
