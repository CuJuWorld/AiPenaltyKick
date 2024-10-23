
## require numpy
## require opencv-python


import cv2
import numpy as np
import APSX_Logger

@APSX_Logger.log_and_time
def extract_frames_at_regular_intervals(video_path: str, n_frames: int) -> list:
    """Extracts frames from a video at regular intervals.

    Args:
        video_path (str): The path to the video file.
        n_frames (int): The number of frames to extract.

    Returns:
        list: A list of extracted frames as NumPy arrays.
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = total_frames // n_frames
    
    frame_indices = range(0, total_frames, frame_interval)

#     extracted_frames = []
#     for current_frame in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#         ret, frame = cap.read()
#         if not ret:
#             break  # Error reading frame

#         extracted_frames.append(frame)

    extracted_frames = []
    current_frame = 0  # Keep track of the current frame number

    while current_frame < total_frames and len(extracted_frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Error reading frame

        if current_frame % frame_interval == 0:
            # RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # extracted_frames.append(RGB_img)
            extracted_frames.append(frame)

        current_frame += 1

    cap.release()
    return extracted_frames

def numpy_image_to_bytes(image_array: np.ndarray) -> bytes:
    """Converts a NumPy image array to a bytes object using OpenCV.

    Args:
        image_array: The NumPy array representing the image.

    Returns:
        bytes: The image data as a bytes object.
    """

    is_success, image_buffer = cv2.imencode(".jpg", image_array)  # Or ".png", etc.
    if not is_success:
        raise RuntimeError("Image encoding failed")

    image_bytes = image_buffer.tobytes()
    return image_bytes
    
@APSX_Logger.log_and_time
def get_frames_from_video(video_file_name, every_nth_frame=10, resize_shape=(512, 512), aspect=None):

    """
    Extracts frames from a video file.

    Args:
        video_file_name (str): The path to the video file.

    Returns:
        list: A list of frames from the video file.
    """
    
    print('Extracting frames from ', video_file_name, every_nth_frame, resize_shape, aspect)

    # laod video in open cv         
    video_file = cv2.VideoCapture(video_file_name)
    num_frames = video_file.get(cv2.CAP_PROP_FRAME_COUNT)
        
    frames = []
    for i in range (int(num_frames)):
        good, frame = video_file.read()
        # skip frames not used         
        if (i % every_nth_frame !=0):
            continue
        if not good:
            break
        
        # crop frame if aspect given      
        if aspect:
            frame = cover_crop_to_aspect(frame, aspect)
        # resize image  
        frame = cv2.resize(frame, (resize_shape[0], resize_shape[1]))
        # correct colour         
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        
    print(f'✅ Extracted {len(frames)} frames from ', video_file_name)
        
    return frames


@APSX_Logger.log_and_time
def get_raw_frames_from_video(video_file_name, every_nth_frame=10):

    """
    Extracts frames from a video file.

    Args:
        video_file_name (str): The path to the video file.

    Returns:
        list: A list of frames from the video file.
    """
    
    print('Extracting frames from ', video_file_name)

    # laod video in open cv         
    video_file = cv2.VideoCapture(video_file_name)
    num_frames = video_file.get(cv2.CAP_PROP_FRAME_COUNT)
    
    new_shape = None
        
    frames = []
    for i in range (int(num_frames)):
        good, frame = video_file.read()
        # skip frames not used         
        if (i % every_nth_frame !=0):
            continue
        if not good:
            break
        # resize image 
        if new_shape is None:
            new_shape = (frame.shape[1]//2, frame.shape[0]//2)
        frame = cv2.resize(frame, new_shape)
        
        # correct colour         
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    print(f'✅ Extracted {len(frames)} frames from ', video_file_name)
        
    return frames

@APSX_Logger.log_and_time
def preprocess_frame(frame, resize_shape=(512, 512)):

    # resize image             
    frame = cv2.resize(frame, resize_shape)
    # correct colour         
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame
        
    
@APSX_Logger.log_and_time
def resize_frame(frame, resize_shape=(512, 512)):

    # resize image             
    frame = cv2.resize(frame, resize_shape)
    return frame
        


@APSX_Logger.log_and_time
def get_frames_at_indexes(video_file_name, indexes):
    
    print(f'Getting {len(indexes)} frames from video {video_file_name}')
    
    video_file = cv2.VideoCapture(video_file_name)
    
    frames = []
    frame_count = 0
    frame_read = True
    
    for index in indexes:
        
        while(frame_read):
            
            frame_read, frame = video_file.read()
            
            if frame_count >= index:
                print('got frame ', index)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                break
                
            frame_count += 1
            
    print(f'✅ Got {len(indexes)} frames from video {video_file_name}')
            
    return frames


def get_indexes_at_times_seconds(time_stamps_seconds, video_file_name):
    
    video = cv2.VideoCapture(video_file_name)
    
    indexes = []
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    for ts in time_stamps_seconds:
        index = int(ts*fps)
        indexes.append(index)
        
    return indexes


def get_frames_at_times(video_file_name, time_stamps_seconds):
    
    indexes = get_indexes_at_times_seconds(time_stamps_seconds, video_file_name)
    frames = get_frames_at_indexes(video_file_name, indexes)
    
    return frames

def cover_crop_protrait_to_aspect(frame, aspect=1):
    
    height, width, _  = frame.shape
    
    new_height = width * aspect
    
    new_height_margin = new_height/2
    height_mid_point = height/2
    
    height_start = int(height_mid_point - new_height_margin)
    height_end =   int(height_mid_point + new_height_margin)
    height_start = max(0, height_start)
    height_end= min(height, height_end)
    
    new_frame = frame[height_start:height_end, : , :]
    return new_frame


def cover_crop_landscape_to_aspect(frame, aspect=1):
    
    height, width, _  = frame.shape
    
    new_width = height / aspect
    
    new_width_margin = new_width/2
    width_mid_point = width/2
    
    width_start = int(width_mid_point - new_width_margin)
    width_end =   int(width_mid_point + new_width_margin)
    width_start = max(0, width_start)
    width_end= min(width, width_end)
    
    new_frame = frame[:, width_start:width_end, :]
    return new_frame


def cover_crop_to_aspect(frame, aspect=1):
    
    height, width, _  = frame.shape
    
    if width > height:
        return cover_crop_landscape_to_aspect(frame, aspect)
    else:
        return cover_crop_protrait_to_aspect(frame, aspect)
