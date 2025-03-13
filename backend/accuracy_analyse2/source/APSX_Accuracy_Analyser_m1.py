# Example Code Snippet (Integrating YOLOv8):

import tensorflow as tf
import cv2
import numpy as np

class Accuracy_Analyser():
    
    def __init__(self, ball_model_path, goal_model_path, bins_model_path, every_nth_frame=5, confidence_threshold=0.5):
        
        # ...existing code...

        # Load YOLOv8 TensorFlow models
        self.ball_model = tf.saved_model.load(ball_model_path)
        self.goal_model = tf.saved_model.load(goal_model_path)
        self.bins_model = tf.saved_model.load(bins_model_path)

        self.confidence_threshold = confidence_threshold
        # ...existing code...

    def predict_goal_position(self, frame, debug):
        '''
        Predict where the goal is in frame, return the normalised bbox
        '''
        # Convert frame to the format expected by TensorFlow
        img = tf.convert_to_tensor(frame, dtype=tf.uint8)
        img = tf.expand_dims(img, 0)  # Add batch dimension

        # Run inference
        infer = self.goal_model.signatures["serving_default"]
        predictions = infer(img)

        # Process the output (adjust based on your model's output format)
        boxes = predictions['output0'].numpy()  # Replace 'output0' with the actual output key
        confidence_scores = boxes[..., 4]  # Assuming confidence is the 5th element

        highest_confidence_index = np.argmax(confidence_scores)
        goal_bbox = boxes[highest_confidence_index][:4]  # Get the bounding box

        # Normalize the bounding box
        height, width, _ = frame.shape
        ymin, xmin, ymax, xmax = goal_bbox
        xmin /= width
        ymin /= height
        xmax /= width
        ymax /= height
        goal_bbox = [ymin, xmin, ymax, xmax]

        if debug:
            # ...existing debug code...
            pass

        return goal_bbox