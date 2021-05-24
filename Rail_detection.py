import cv2
import numpy as np
import time
import line
import utils
from PIL import Image

def processing(img,M,Minv,left_line,right_line):
    prev_time = time.time()
    img = Image.fromarray(img)
    undist = img
    #get the thresholded binary image
    img = np.array(img)
    blur_img = cv2.GaussianBlur(img,(3,3),0)
    Sobel_x_thresh = utils.abs_sobel_thresh(blur_img, orient='x', thresh_min=90 ,thresh_max=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(Sobel_x_thresh, kernel, iterations = 2)
    #perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(dilated, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    #perform detection
    if left_line.detected and right_line.detected:
        left_fit, right_fit = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit= utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # #draw the detected laneline and the information
    undist = Image.fromarray(img)
    area_img = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    area_img = np.array(area_img)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    return area_img,thresholded_wraped
#----------------------------

left_line = line.Line()
right_line = line.Line()
M,Minv = utils.get_M_Minverse()

#____________________________

cap = cv2.VideoCapture('RailTest3.mp4')
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
    Pts = np.float32([[680,1080],[920,500],[1245,1080],[1050,500]])
    res,t1 = processing(frame,M,Minv,left_line,right_line)
    Res,area = np.asarray(res)
    cv2.imwrite('./resIMG.jpg', Res)
    resize_area = cv2.resize(area,(1080,720))
    resize_Res = cv2.resize(Res,(1080,720))
    cv2.imshow("area",resize_area)
    cv2.imshow("Rail detection",resize_Res)
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

    