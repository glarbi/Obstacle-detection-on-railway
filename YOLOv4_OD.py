import time
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np

input_size = 416
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


cap = cv2.VideoCapture('RailTest1.mp4')
# Check if camera opened successfully
fps = int(cap.get(cv2.CAP_PROP_FPS))
if (cap.isOpened()== False):
  print("Error opening video stream or file")
# Read until video is completed

while(cap.isOpened()):

  # Capture frame-by-frame

  ret, frame = cap.read()
  if ret == True:  

    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.35
        )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(frame, pred_bbox)
    exec_time = time.time() - start_time
    fps = 1.0 / (time.time() - start_time)
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    print("FPS: %.2f" % fps)
    result = np.asarray(image)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    cv2.imwrite('./YOLOv4Result.jpg', result)
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