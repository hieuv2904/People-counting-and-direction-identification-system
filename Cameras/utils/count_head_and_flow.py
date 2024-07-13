import PIL.Image as Image
import numpy as np
import os
import cv2
from google.colab.patches import cv2_imshow

def cal_contour_pixel(flow, contours):
  mask = np.zeros(flow.shape[:2], dtype=np.uint8)
  cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
  masked_image = cv2.bitwise_and(flow, flow, mask=mask)
  sum_pixels = np.sum(masked_image)
  return sum_pixels

def count_head_and_flow(flow_2, flow_4, flow_6, flow_8, density_map, original_image):
  '''
  @flow_2: np array of flow up
  @flow_4: np array of flow left
  @flow_6: np array of flow right
  @flow_8: np array of flow down
  @density_map: np array of total density map
  @original_image: current frame
  '''
  flow_2 = np.array(flow_2)
  flow_4 = np.array(flow_4)
  flow_6 = np.array(flow_6)
  flow_8 = np.array(flow_8)

  density_map = np.array(density_map)
  

  # cv_image = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)




  #draw on original image
  color_maps = {"up": (255, 0, 0), "down": (0, 255, 0), "left": (0, 0, 255), "right": (125, 0, 125)} #bgr
  # density_map = cv2.cvtColor(density_map, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(density_map, 60, 255, cv2.THRESH_BINARY)

  #erosion and dilatesion
  kernel = np.ones((5,5),np.uint8)

  thresh = cv2.erode(thresh,kernel,iterations = 2)
  thresh = cv2.dilate(thresh,kernel,iterations = 2)

  # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
  contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
  original_image_copy = np.copy(original_image)
  num_circles = len(contours)

  # print(contours[0])
  for i in contours:
      up_identical_index = cal_contour_pixel(flow_2, i)
      left_identical_index = cal_contour_pixel(flow_4, i)
      right_identical_index = cal_contour_pixel(flow_6, i)
      down_identical_index = cal_contour_pixel(flow_8, i)

      identical_index = up_identical_index
      direction = 'up'

      if identical_index < left_identical_index:
        direction = 'left'
        identical_index = left_identical_index
      if identical_index < right_identical_index:
        direction = 'right'
        identical_index = right_identical_index
      if identical_index < down_identical_index:
        direction = 'down'
        identical_index = down_identical_index

      M = cv2.moments(i)
      if M['m00'] != 0:
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          cv2.circle(original_image_copy, (cx, cy), 7, color_maps[direction], -1)



  cv2.putText(original_image_copy, f'Count: {num_circles}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
  cv2.putText(original_image_copy, 'Up', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color_maps['up'], 2, cv2.LINE_AA)
  cv2.putText(original_image_copy, "Down", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color_maps['down'], 2, cv2.LINE_AA)
  cv2.putText(original_image_copy, "Left", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color_maps['left'], 2, cv2.LINE_AA)
  cv2.putText(original_image_copy, 'Right', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_maps['right'], 2, cv2.LINE_AA)

  return original_image_copy