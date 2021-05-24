import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import line
import utils

#-----------------------------
def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=90 ,thresh_max=280)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(160, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 210))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    return threshholded
def processing(img,M,Minv,left_line,right_line):
    prev_time = time.time()
    img = Image.fromarray(img)
    undist = img
    #get the thresholded binary image
    img = np.array(img)
    thresholded = thresholding(img)
    #perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    #perform detection
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # #draw the detected laneline and the information
    undist = Image.fromarray(img)
    area_img, gre1 = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    curvature,pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    area_img = np.array(area_img)
    result = utils.draw_values(area_img,curvature,pos_from_center)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    return result,thresholded_wraped
#----------------------------

left_line = line.Line()
right_line = line.Line()
M,Minv = utils.get_M_Minv()

#____________________________

cap = cv2.VideoCapture('RailTest2.mp4')
# Check if camera opened successfully
fps = int(cap.get(cv2.CAP_PROP_FPS))
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed

while(cap.isOpened()):

  # Capture frame-by-frame

  ret, frame = cap.read()
  if ret == True:
    #[600,1080],[850,300],[1600,1080],[1000,300]-----> RailTest1.mp4
    #[610,360],[768,360],[900,720],[560,720]--------->RailTest2.mp4
    Pts = np.float32([[680,1080],[920,520],[1245,1080],[995,520]])
    for i in range (0,4):
        cv2.circle(frame,(Pts[i][0],Pts[i][1]),5,(0,0,255),cv2.FILLED)
    res,t1 = processing(frame,M,Minv,left_line,right_line)
    Res = np.asarray(res)
    cv2.imwrite('./resIMG.jpg', Res)
    resize_frame = cv2.resize(frame,(640,360))
    resize_Res = cv2.resize(Res,(640,360))
    cv2.imshow("Run",resize_Res)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()

    