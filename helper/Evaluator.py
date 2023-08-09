def evaluate_accuracy(output_path, groundtruth_path, focus_track_id, threshold):
    with open(output_path, 'r') as file1, open(groundtruth_path, 'r') as file2:
      lines1 = file1.readlines()
      lines2 = file2.readlines()

    total_frame_count = len(lines2)
    detected_frame_count = 0

    for line1 in lines1:
      values1 = line1.split()
      track_id = int(values1[1])

      if (track_id != focus_track_id): continue # go next if not the focused object

      frame_id = int(values1[0])
      left_1 = int(values1[2])
      top_1 = int(values1[3])
      weight_1 = int(values1[4])
      height_1 = int(values1[5])

      if frame_id < len(lines2):
        values2 = lines2[frame_id].split()
        left_2 = int(values2[0])
        top_2 = int(values2[1])
        weight_2 = int(values2[2])
        height_2 = int(values2[3])

        if is_contained_with_threshold(
            left_1, top_1, weight_1, height_1, 
            left_2, top_2, weight_2, height_2,
            threshold
            ):
          
          detected_frame_count += 1

    return detected_frame_count / total_frame_count

def is_contained_with_threshold(
                left_1, top_1, weight_1, height_1, 
                left_2, top_2, weight_2, height_2,
                threshold):
    x_left_1 = left_1
    y_top_1 = top_1
    x_right_1 = x_left_1 + weight_1
    y_bottom_1 = y_top_1 - height_1

    x_left_2 = left_2
    y_top_2 = top_2
    x_right_2 = x_left_2 + weight_2
    y_bottom_2 = y_top_2 - height_2

    # print(x_left_2, y_top_2, x_right_2, y_bottom_2)

    x_left = max(x_left_1, x_left_2)
    y_bottom = max(y_bottom_1, y_bottom_2)
    x_right = min(x_right_1, x_right_2)
    y_top = min(y_top_1, y_top_2)
    
    # print(x_left, y_top, x_right, y_bottom)

    if x_right < x_left or y_top < y_bottom:
      # No intersection, return 0
      return False
    else:
      intersection_area = (x_right - x_left) * (y_top - y_bottom)
      rectangle_2_area = weight_2 * height_2

      return intersection_area > threshold * rectangle_2_area
    

# ---------- test case ---------
# print(is_contained_with_threshold(
#                 1, 3, 1, 2, 
#                 0, 2, 2, 1,
#                 -1))


accuracy = evaluate_accuracy('./output/output.txt', './benchmark/CarScale/groundtruth_rect.txt', 1, 0.8)
print("accuracy: ", accuracy)