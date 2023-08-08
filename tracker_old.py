# ---------------------------
# author: Mingzhe Wang
# date: 2023 Aug 8
# description:
#   this is a main function of the visual object tracking project
#   usgae is as the following:
#     `python3 tracker.py camera``
#     `python3 tracker.py video 'input_filename'`
#     `python3 tracker.py benchmark 'test_case_name'`
# ---------------------------

import cv2 # python opencv for image processing
from ultralytics import YOLO # yolo v8 for object detection
from deep_sort_realtime.deepsort_tracker import DeepSort # deepsort for object tracking
from helper.VideoWriter import VideoWriter
import numpy as np
import sys
import os
import datetime

# set path configuration
current_path = os.getcwd()
mode = sys.argv[1]
if mode == 'video':
  input_filename = sys.argv[2]
elif mode == 'benchmark':
  test_case_name = sys.argv[2]
  test_case_path = current_path + '/benchmark/' + test_case_name + '/img'

# layout and output configuration
color_size = 20 # max number of objects in the frame
colors = [list(np.random.random(size=3) * 256) for _ in range(color_size)] # hold different rgb colors for each object
MIN_CONFIDENCE = 0.6 # min confidence for object detection
MAX_AGE = 30 # max age for object tracking

# create detector, tracker, video_reader, video writer
detector = YOLO("yolov8n.pt")
# detector.train(data='coco.yaml', epochs=100, imgsz=640) # Train the model
tracker = DeepSort(max_age=MAX_AGE)
if mode == 'video':
  video_reader = cv2.VideoCapture(current_path + '/video/'  + input_filename)
elif mode == 'benchmark':
  video_reader = cv2.VideoCapture(test_case_path + "/%04d.jpg", cv2.CAP_IMAGES)
else:
  video_reader = cv2.VideoCapture(0)
video_writer = VideoWriter(video_reader, current_path + "/output/output.mp4")

# start the tracking
frame_counter = 0
with open(current_path + '/output/output.txt', 'w') as file:
  while True:
    start_time = datetime.datetime.now()

    ### --- video reading: video -> frame --- ###
    retval, frame = video_reader.read()
    if not retval: break

    ### --- dectecting: frame -> list of ([left,top,w,h], confidence, detection_class) --- ###
    detections = detector(frame)[0].boxes.data.tolist()
    results = []
    # modify the data structure in detections
    for i in range(len(detections)):
      dectection = detections[i]
      confidence = float(dectection[4])
      if confidence < MIN_CONFIDENCE: continue
      left, top, right, bottom = \
        int(dectection[0]), int(dectection[1]), int(dectection[2]), int(dectection[3])
      w = right - left
      h = bottom - top
      detection_class = int(dectection[5])
      results.append(([left,top,w,h], confidence, detection_class))

    ### --- tracking: list of ([left,top,w,h], confidence, detection_class), frame -> tracks (i.e. track_id, ltrb) --- ###
    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
      if not track.is_confirmed(): continue
      # get track id
      track_id = int(track.track_id)
      # get bounding box
      ltrb = track.to_ltrb()
      xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
      # draw track id and bounding box
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors[track_id % color_size], 2)
      cv2.rectangle(frame, (xmin - 1, ymin - 30), (xmin + 31, ymin), colors[track_id % color_size], -1)
      cv2.putText(frame, str(track_id), (xmin + 5, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      # output (frame, track_id, left, top, weight, height) to file
      file.write(f"{frame_counter}  {track_id}  {xmin}  {ymin}  {xmax - xmin}  {ymax - ymin}\n")

    ### --- output: display and store output frame withtracking result --- ###
    end_time = datetime.datetime.now()
    fps = f"FPS: {1 / (end_time - start_time).total_seconds():.2f}"
    cv2.putText(frame, fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
    cv2.imshow("Output", frame)
    video_writer.write(frame)
  
    ### --- controller: quit using `q` button --- ###
    if cv2.waitKey(1) == ord("q"): break
      
    ### --- update the frame info --- ###
    frame_counter += 1
    
# release all components
video_reader.release()
video_writer.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)