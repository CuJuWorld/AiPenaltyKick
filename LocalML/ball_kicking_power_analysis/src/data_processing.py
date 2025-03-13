import cv2
import pandas as pd
import numpy as np
import os

def load_video(video_path):
    """
    Load video from the specified path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    video_capture = cv2.VideoCapture(video_path)
    return video_capture

def extract_frames(video_capture, every_nth_frame=5):
    """
    Extract frames from the video capture object at specified intervals.
    """
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % every_nth_frame == 0:
            frames.append(frame)
        frame_count += 1
    
    video_capture.release()
    return frames

def process_video_data(video_path, every_nth_frame=5):
    """
    Load and process video data to extract frames.
    """
    video_capture = load_video(video_path)
    frames = extract_frames(video_capture, every_nth_frame)
    return frames

def save_features_to_csv(features, output_path):
    """
    Save extracted features to a CSV file.
    """
    df = pd.DataFrame(features)
    df.to_csv(output_path, index=False)

def load_labels(label_path):
    """
    Load kicking power labels from a CSV file.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    
    labels = pd.read_csv(label_path)
    return labels

def process_data(video_path, label_path, output_features_path, every_nth_frame=5):
    """
    Process video data and save extracted features and labels.
    """
    frames = process_video_data(video_path, every_nth_frame)
    # Placeholder for feature extraction logic
    features = []  # Replace with actual feature extraction from frames
    
    save_features_to_csv(features, output_features_path)
    labels = load_labels(label_path)
    
    return features, labels