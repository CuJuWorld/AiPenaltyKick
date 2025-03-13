import cv2
import numpy as np
import pandas as pd

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Example feature extraction: calculate frame mean color
        mean_color = cv2.mean(frame)[:3]  # Get mean color in BGR
        features.append(mean_color)

    cap.release()
    return np.array(features)

def save_features_to_csv(features, output_path):
    df = pd.DataFrame(features, columns=['B', 'G', 'R'])
    df.to_csv(output_path, index=False)

def main():
    video_path = '../data/raw/video_data.mp4'
    output_path = '../data/processed/features.csv'
    
    features = extract_features(video_path)
    save_features_to_csv(features, output_path)

if __name__ == "__main__":
    main()