import cv2 # python opencv for image processing
from ultralytics import YOLO # yolo v8 for object detection
from deep_sort_realtime.deepsort_tracker import DeepSort # deepsort for object tracking
from helper.VideoWriter import VideoWriter
import numpy as np
import sys
import os
import datetime

class ObjectTracker():
    def __init__(self, color_size = 20, MIN_CONFIDENCE = 0.6, MAX_AGE = 30):
        # set path configuration
        self.current_path = os.getcwd()
        self.mode = sys.argv[1]
        if self.mode == 'video':
          self.input_filename = sys.argv[2]
        elif self.mode == 'benchmark':
          self.test_case_name = sys.argv[2]
          self.test_case_path = self.current_path + '/benchmark/' + self.test_case_name + '/img'

        # layout and output configuration
        self.color_size = color_size # max number of objects in the frame
        self.MIN_CONFIDENCE = MIN_CONFIDENCE# min confidence for object detection
        self.MAX_AGE = MAX_AGE # max age for object tracking
        self.colors = [list(np.random.random(size=3) * 256) for _ in range(self.color_size)] # hold different rgb colors for each object

        # create detector, tracker, video_reader, video writer
        self.detector = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=MAX_AGE)
        if self.mode == 'video':
          self.video_reader = cv2.VideoCapture(self.current_path + '/video/'  + self.input_filename)
        elif self.mode == 'benchmark':
          self.video_reader = cv2.VideoCapture(self.test_case_path + "/%04d.jpg", cv2.CAP_IMAGES)
        else:
          self.video_reader = cv2.VideoCapture(0)
        self.video_writer = VideoWriter(self.video_reader, self.current_path + "/output/output.mp4")

    def train(self, data, epochs, imgsz):
        # e.g. data = 'coco.yaml', epochs=100, imgsz=640
        self.detector.train(data=data, epochs=epochs, imgsz=imgsz) # Train the model

    def run(self):
        # start the tracking
        frame_counter = 0
        with open(self.current_path + '/output/output.txt', 'w') as file:
          while True:
            start_time = datetime.datetime.now()

            ### --- video reading: video -> frame --- ###
            retval, frame = self.video_reader.read()
            if not retval: break

            ### --- dectecting: frame -> list of ([left,top,w,h], confidence, detection_class) --- ###
            detections = self.detector(frame)[0].boxes.data.tolist()
            results = []
            # modify the data structure in detections
            for i in range(len(detections)):
              dectection = detections[i]
              confidence = float(dectection[4])
              if confidence < self.MIN_CONFIDENCE: continue
              left, top, right, bottom = \
                int(dectection[0]), int(dectection[1]), int(dectection[2]), int(dectection[3])
              w = right - left
              h = bottom - top
              detection_class = int(dectection[5])
              results.append(([left,top,w,h], confidence, detection_class))

            ### --- tracking: list of ([left,top,w,h], confidence, detection_class), frame -> tracks (i.e. track_id, ltrb) --- ###
            tracks = self.tracker.update_tracks(results, frame=frame)
            for track in tracks:
              if not track.is_confirmed(): continue
              # get track id
              track_id = int(track.track_id)
              # get bounding box
              ltrb = track.to_ltrb()
              xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
              # draw track id and bounding box
              cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.colors[track_id % self.color_size], 2)
              cv2.rectangle(frame, (xmin - 1, ymin - 30), (xmin + 31, ymin), self.colors[track_id % self.color_size], -1)
              cv2.putText(frame, str(track_id), (xmin + 5, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
              # output (frame_id, track_id, left, top, weight, height) to file
              file.write(f"{frame_counter}  {track_id}  {xmin}  {ymin}  {xmax - xmin}  {ymax - ymin}\n")

            ### --- output: display and store output frame withtracking result --- ###
            end_time = datetime.datetime.now()
            fps = f"FPS: {1 / (end_time - start_time).total_seconds():.2f}"
            cv2.putText(frame, fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
            cv2.imshow("Output", frame)
            self.video_writer.write(frame)

            ### --- controller: quit using `q` button --- ###
            if cv2.waitKey(1) == ord("q"): break
              
            ### --- update the frame info --- ###
            frame_counter += 1

        # release all components
        self.video_reader.release()
        self.video_writer.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
