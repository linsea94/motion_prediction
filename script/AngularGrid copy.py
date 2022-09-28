import numpy as np
import math
import matplotlib.pyplot as plt

class pt:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def distance(target, other):
    dist = (target.x - other.x)**2 + (target.y-other.y)**2
    return np.round(math.sqrt(dist), 4)
  def direction(target, other, num_of_pieces=8):
    a = other.y - target.y
    b = other.x - target.x
    theta = (round(np.arctan2(a, b)/math.pi*180,4))
    if theta < 0:
      theta = theta + 360
    a_pieces = 360 / num_of_pieces
    dir = int(theta / a_pieces)
    return dir

def get_pt(lst):
  point = pt(lst[2],lst[3])
  return point

def get_AngularGrid(other_lst, target_pt, num_of_pieces=72, max_dist=10):
  surrounding = np.ones(num_of_pieces)
  for i in range(0,len(other_lst)):
    other = other_lst.iloc[i]
    other_pt = pt(other[2],other[3])
    dist = pt.distance(target_pt, other_pt)
    dist = np.round(dist / max_dist, 4)
    if dist > 1:
      continue
    else:
      pass
    dir = pt.direction(target_pt, other_pt, num_of_pieces)
    if dist < surrounding[dir]:
      surrounding[dir] = dist
  return surrounding

def draw_AngularGrid(surrounding, num_of_pieces = 72):
  cmap = plt.get_cmap("Reds")
  color = cmap(np.array(np.abs(surrounding-1)))
  img = plt.pie(np.ones(num_of_pieces),
          explode = surrounding,
          labels = surrounding,
          colors = color,
          textprops = {"fontsize" : 10})
  plt.show()


def draw_scatter(target, other_lst):
  plt.scatter(target[2], target[3])
  plt.text(target[2],target[3], target[1])

  for other in other_lst:
    plt.scatter(other[2],other[3])
    plt.text(other[2],other[3], other[1])