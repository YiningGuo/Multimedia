"""Provides distortion for images, this module can be used to create
   artificial dataset.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Distortion(object):

  def __init__(self):
    self.fixed_size = 28
    self.start_idx = 0 # the position to start distortion
    self.end_idx = 28 # the position to end distortion
    self.middle_idx = (self.end_idx - self.start_idx) // 2
    self.twist_func = {"top": self.twist_to_top,
                       "buttom": self.twist_to_buttom,
                       "left": self.twist_to_left,
                       "right": self.twist_to_right}

  def twist_to_top(self, row, col, distort_rate):
    new_col = col
    new_row = (row + distort_rate) % self.fixed_size
    return new_row, new_col

  def twist_to_buttom(self, row, col, distort_rate):
    new_col = col
    new_row = (row - distort_rate) % self.fixed_size
    return new_row, new_col

  def twist_to_left(self, row, col, distort_rate):
    new_col = (col + distort_rate) % self.fixed_size
    new_row = row
    return new_row, new_col

  def twist_to_right(self, row, col, distort_rate):
    new_col = (col - distort_rate) % self.fixed_size
    new_row = row
    return new_row, new_col

  def generate_maps(self, update_func):
    map_size = (self.fixed_size, self.fixed_size)
    map_x = np.zeros(map_size, np.float32)
    map_y = np.zeros(map_size, np.float32)
    for row in range(self.start_idx, self.end_idx):
      distort_rate = 1
      count = 0
      for col in range(self.start_idx, self.end_idx):
        new_row, new_col = update_func(row, col, distort_rate)
        map_x.itemset((row, col), new_col)
        map_y.itemset((row, col), new_row)
        if col < self.middle_idx and count == 2:
          distort_rate += 1
          count = 0
        elif col >= self.middle_idx and count == 2:
          distort_rate -= 1
          count = 0
        else:
          count += 1
    return map_x, map_y

  def process(self, image_name, direction):
    img = cv2.imread(image_name)
    distort = Distortion()
    map_x, map_y = distort.generate_maps(self.twist_func[direction])
    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    # show the result image
    plt.imshow(result)
    plt.show()
