import cv2

def VideoWriter(video_reader, output_filename):
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return video_writer