
import math

import numpy as np

import matplotlib.pyplot as plt
from google.cloud import storage

import APSX_Predictor
import APSX_Object_Tracker
import APSX_Video_Processor
import APSX_Plotter
import APSX_GCS_utils

import APSX_Vertex_Predictor  

import APSX_Event_Info_Helper

event_info_helper = APSX_Event_Info_Helper.APSX_Event_Info_Helper()


class Power_Analyser():
    
    def __init__(self, model_path="model.tflite", every_nth_frame=4, confidence_threshold=0.5):
        
        self.ball_predictor = APSX_Predictor.Predictor(model_path)
        # self.ball_predictor = APSX_Vertex_Predictor.Predictor('7748480542286282752')

        
        self.every_nth_frame = every_nth_frame
        self.confidence_threshold = confidence_threshold
        self.seconds_between_frames = 1/240
        self.frame_shape = (192, 192)
        
        print('✅ Loaded Power_Analyser')

    
    def analyse(self, gs_file_name, event_id, debug=False):
        
        self.event_id = event_id
        self.asset_bucket  = event_info_helper.get_misc_data_bucket(event_id)
        
        # download video
        local_file_name = APSX_GCS_utils.download_from_gcs(gs_file_name)
        print('local file: ', local_file_name)
        
        # get video frames         
        video_frames = APSX_Video_Processor.get_frames_from_video(local_file_name, self.every_nth_frame, self.frame_shape)
        
        # predict on each frame
        predictions = self.ball_predictor.predict_on_frames(video_frames, threshold=self.confidence_threshold)
        
        # get ball track from predictions         
        ball_track = self.get_ball_track(predictions)
        
        # calculate speed
        action_id = APSX_GCS_utils.APSX_File_Name(gs_file_name).action_id
        speed, metadata = self.calculate_speed(ball_track, video_frames, action_id, debug=debug)
        
        return speed, metadata
    
    @staticmethod
    def front_pad_zeros(array, n):
        return [0]*n +array
    
    
    def calculate_speed(self, ball_track, video_frames, action_id, debug=False):
        
        if ball_track is None:
            return (0,{'raw_score':0,
                    'power_plot': None,
                   'kick_event_seconds':0,
                   'prediction_count': len(video_frames)})

        midpoints = ball_track.midpoints
        distances = ball_track.distances
        
        speeds = [d / (self.every_nth_frame * self.seconds_between_frames) for d in  distances]
        
        # TODO very messy fix for when ball is outside the frame initially         
        speeds = self.front_pad_zeros(speeds, ball_track.start_frame_num)
        print('start frame num', ball_track.start_frame_num)
        print('speeds', speeds)
        
        if debug:
            plt.title('distances')
            plt.plot(distances)
            plt.show()
        if debug:
            plt.title('Speeds')
            plt.plot(speeds)
            plt.show()
            
        
 
        
        kick_event_index, kick_event_seconds = self.get_kick_event(speeds)
        
        # TODO very messy fix for when ball is outside the frame initially         
        if speeds[0] == 0:
            kick_event_index = 0
        
        print('kick_event_index, kick_event_seconds', kick_event_index, kick_event_seconds)
        
        if debug:
            plt.title('Kick impact frame')
            plt.imshow(video_frames[kick_event_index])
            plt.show()
            
            
        power_plot_file_name = 'power_plot.png'
        power_plot_gcs_file_name = f'{self.asset_bucket}/{action_id}/{power_plot_file_name}'
        
        APSX_Plotter.plot_ball_bboxes(video_frames, ball_track, kick_event_index, file_name=power_plot_file_name)
        APSX_GCS_utils.upload_to_gcs(power_plot_file_name, power_plot_gcs_file_name)
        
        raw_speed = max(speeds)
        speed_score = int(self.scale_and_limit_value(raw_speed, 0, 10, 50, 99))
        metadata = {'raw_score':raw_speed,
                    'power_plot': power_plot_gcs_file_name,
                   'kick_event_seconds':kick_event_seconds,
                   'prediction_count': len(video_frames)}
        return speed_score, metadata
    
    def get_ball_track(self, bbox_predictions):
        
        self.ball_tracker = APSX_Object_Tracker.ObjectTracker(max_distance=2000, max_missing_frames=15)
        self.ball_tracker.analyse(bbox_predictions)
        ball_track = self.ball_tracker.get_longest_track()
        
        return self.ball_tracker.get_longest_track()
    
    def get_kick_event(self, speeds):
                
        kick_event_index = max(np.argmax(np.array(speeds))-1, 0)
        
        if kick_event_index != 0:
            kick_event_seconds = kick_event_index * self.every_nth_frame * self.seconds_between_frames
        else: 
            kick_event_seconds = 0
            
        return kick_event_index, kick_event_seconds
        
    
    def scale_and_limit_value(self, value, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        scaled_value = (((value - old_min) * new_range) / old_range) + new_min
        limited = min(max(scaled_value, new_min), new_max)
        return limited
