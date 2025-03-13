
import APSX_Logger

import tensorflow as tf

class Predictor():
    '''
    Class managing the tflite object detection model interaction
    '''
    
    def __init__(self, model_path="model.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        
    def predict(self, X):
        # predict on a single frame         
            
        # load data             
        self.interpreter.set_tensor(self.input_details[0]['index'], [X])

        # Run the inference
        self.interpreter.invoke()

        # get output     
        confidences = self.interpreter.get_tensor(self.output_details[2]['index'])
        bboxs = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if (len(confidences.shape) > 1) and confidences.shape[0] == 1:
            return confidences[0], bboxs[0]
        
        return confidences, bboxs
    
    @APSX_Logger.log_and_time
    def predict_on_frames(self, video_frames, threshold=0.5, debug=False):
        # just get bounding box predictions from video frames
        # given a threshold
        
        print(f'Predicting on {len(video_frames)} frames with {threshold} threshold')

        
        predictions = []
        
        for frame in video_frames:
            
            # get prediction  
            confidences, bboxes = self.predict(frame)
            mask = (confidences <= 1) & (confidences > threshold)
            if debug:
                print('confidences: ',confidences)
                print('bboxes: ', bboxes)
                pred_count = sum(mask)
                print(f'{pred_count} boxes over threshold')
                
            predictions.append(bboxes[mask])
            
        print(f'✅ Predicted on {len(video_frames)} frames with {threshold} threshold, {len(predictions)} made')


        return predictions
