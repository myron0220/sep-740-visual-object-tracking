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

from tracker import ObjectTracker

object_tracker = ObjectTracker(color_size = 20, MIN_CONFIDENCE = 0.6, MAX_AGE = 30)

# object_tracker.train()

object_tracker.run()
