"""Simple implementation of Local Binary Pattern:
   a texture descriptor
   corresponding library: skimage.feature.local_binary_pattern
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

class LBP(object):

  def __init__(self):
    self.image = None
    self.image_width = 200
    self.image_height = 200

  def _get_neighbour_threshold(self, y, x, center):
    """Calculate the threshold for a neighbour pixel
    Args:
      y: int, the y axis of neighbour pixel
      x: int, the x axis of neighbour pixel
      center: the grayscale value of center
    """
    if x < 0 or y < 0 or x >= self.image_width or y >= self.image_height:
      return 1
    else:
      return int(center > self.image[y][x])

  def cal_pixel_value(self, x, y):
    """Calculate the LBP value for center pixel in a n*n window
    Args:
      x: int, the x axis of center pixel
      y: int, the y axis of center pixel
    """
    n = 5 # n can only be odd number
    center_value = self.image[y][x]
    binary_vec = [] # threshold of neighbours compared with center

    max_x = x + (n // 2)
    max_y = y + (n // 2)
    min_x = x - (n // 2)
    min_y = y - (n // 2)

    # right border, start from top-right corner
    for y in range(min_y, max_y + 1):
      binary_vec.append(self._get_neighbour_threshold(y, max_x, center_value))

    # buttom border, move clockwise
    for x in range(max_x - 1, min_x - 1, -1):
      binary_vec.append(self._get_neighbour_threshold(max_y, x, center_value))

    # left border
    for y in range(max_y - 1, min_y - 1, -1):
      binary_vec.append(self._get_neighbour_threshold(y, min_x, center_value))

    # top border
    for x in range(min_x + 1, max_x):
      binary_vec.append(self._get_neighbour_threshold(min_y, x, center_value))

    # calculate center lbp value
    value = 0
    for i in range(len(binary_vec)):
      value += binary_vec[i] * (2 ** i)
    return value

  def move_window(self):
    lbp_img = np.zeros(self.image.shape)
    for y in range(self.image_height):
      for x in range(self.image_width):
        lbp_img[y][x] = self.cal_pixel_value(x, y)
    return lbp_img

  def cal_histogram(self, lbp_img):
    # the number of possible value is 256 if n=3, where n is the size of window
    hist_bin = np.histogram(lbp_img, range(0, 65536, 500))
    hist = hist_bin[0] / np.linalg.norm(hist_bin[0]) # normalise
    return hist

  def read_image(self, image_name):
    self.image = cv2.imread(image_name, 0)
    if self.image.shape[0] > 200:
      self.image = cv2.resize(self.image, (200,200))
    else:
      self.image_height, self.image_width = self.image.shape[:2]

  def compare_hists(self, hist1, hist2, eps = 1e-10):
    """Return a distance of the two histograms.
    Args:
      eps: avoid divide by 0
    """
    #Chi-Squared: large difference contribute less weight.
    # distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
	# 	       for (a, b) in zip(hist1, hist2)])

    # Intersection
    # distance = np.mean([min(a, b) for (a, b) in zip(hist1, hist2)])

    # Correlation, similarity from 0-1
    distance = np.corrcoef(hist1, hist2)[0][1]
    return distance

  def process(self, image_name):
    self.read_image(image_name)
    result = self.move_window()
    hist = self.cal_histogram(result)

    # show the result image
    # plt.subplot(131)
    # plt.imshow(cv2.imread(image_name))
    # plt.title("origin")
    # plt.axis('off')
    #
    # plt.subplot(132)
    # plt.imshow(result, cmap="gray")
    # plt.title("lbp image")
    # plt.axis('off')
    #
    # plt.subplot(133)
    # plt.ylim((0, 0.6))
    # plt.bar(range(1, 132), hist, 1)
    # plt.title("lbp histogram")
    #
    # plt.show()
    return hist

if __name__ == '__main__':
  l = LBP()
  hist1 = l.process("0_6.jpg")
  hist2 = l.process("0_0.jpg")
  sim = l.compare_hists(hist1, hist2)
  print(sim)
