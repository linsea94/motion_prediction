import os
from re import X
import numpy as np

# OPENTRAJ_ROOT = "/home/linsea/motion_ws/src/motion_prediction/data/ETH_v1/"
# TRAJ = (x, y)

# def world2image(traj_w, H_inv):    
#     # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
#     traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
#     # to camera frame
#     traj_cam = np.matmul(H_inv, traj_homog)  
#     # to pixel coords
#     traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
#     return traj_uvz[:, :2].astype(int)    

# H = (np.loadtxt(os.path.join(OPENTRAJ_ROOT, "H.txt")))
# H_inv = np.linalg.inv(H)
# world2image({TRAJ}, H_inv)  # TRAJ: Tx2 numpy array

def traj2pixal(traj,path_H):
    h = (np.loadtxt(os.path.join(path_H)))
    h_inv = np.linalg.inv(h)
    traj_homog = np.hstack((traj, np.ones((traj.shape[0], 1)))).T
    traj_cam = np.matmul(h_inv, traj_homog)  
    traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
    return traj_uvz[:, :2].astype(int)    
