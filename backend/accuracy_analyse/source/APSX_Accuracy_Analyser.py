
import math
import numpy as np
import matplotlib.pyplot as plt

import APSX_Object_Tracker 
import APSX_Predictor  
import APSX_Video_Processor
import APSX_GCS_utils
import APSX_Plotter

import APSX_Event_Info_Helper

event_info_helper = APSX_Event_Info_Helper.APSX_Event_Info_Helper()




class Accuracy_Analyser():
    
    def __init__(self, ball_model_path, goal_model_path, bins_model_path, every_nth_frame=5, confidence_threshold=0.5):
        
        # ball_model_path = "goal_ball_top-_bins_low_latency_model-6531246403547561984_tflite_2023-09-23T08_35_54.085987Z_model.tflite"
        self.ball_predictor = APSX_Predictor.Predictor(ball_model_path)
        self.goal_predictor = APSX_Predictor.Predictor(goal_model_path)
        self.bins_predictor = APSX_Predictor.Predictor(bins_model_path)
        print('✅ Models loaded')
        
        self.ball_tracker = None
        
        self.every_nth_frame = every_nth_frame
        
        self.confidence_threshold = confidence_threshold
        
        self.seconds_between_frames = 1/240
        self.frame_shape = (192, 192)

    
    def analyse(self, gs_file_name, targets=[[0.1,0.1], [0.9,0.1]], event_id = None, debug=False):
        
        self.event_id = event_id
        self.asset_bucket  = event_info_helper.get_misc_data_bucket(event_id)

        
        # download video
        local_file_name = APSX_GCS_utils.download_from_gcs(gs_file_name)
        print('local file: ', local_file_name)
        
        # get video frames         
        video_frames = APSX_Video_Processor.get_frames_from_video(local_file_name, self.every_nth_frame, self.frame_shape)
        # video_frames = APSX_Video_Processor.get_raw_frames_from_video(local_file_name, self.every_nth_frame)

        # predict on each frame
        predictions = self.ball_predictor.predict_on_frames(video_frames, threshold=self.confidence_threshold)
        
        # get longest track from predictions         
        self.ball_tracker = APSX_Object_Tracker.ObjectTracker(max_distance=0.4, max_missing_frames=20)
        self.ball_tracker.analyse(predictions)
        print(f'objects found tracked', len(self.ball_tracker.tracks))
        ball_track = self.ball_tracker.get_longest_track()
        
        if debug:
            print(predictions)
        
        # get position of goal

        # goal_frame = APSX_Video_Processor.resize_frame(video_frames[len(video_frames)//2], 
        #                                                    resize_shape=(512, 512))
        # goal_frame = goal_frame.astype(np.uint8)
        goal_frame = video_frames[len(video_frames)//2]
        goal_position = self.predict_goal_position(goal_frame, debug=debug)
        
        bin_positions = self.predict_bin_positions(goal_frame, debug=debug)
        # goal_position = [0,1,2,4]        
        # calculate accuracy
        action_id = APSX_GCS_utils.APSX_File_Name(gs_file_name).action_id
        accuracy, metadata = self.calculate_accuracy(ball_track, targets, goal_position, bin_positions, action_id, delta_direction_threshold=0.6,
                                                     debug=debug, debug_frames=video_frames)
        
        metadata['prediction_count'] = len(video_frames)
        
        return accuracy, metadata
    
    @staticmethod
    def top_n_argmax(array, n):
        return np.argpartition(array, -2)[-2:]

    def predict_bin_positions(self, frame, debug):
        '''
        Predict where the bins are in frame, return the normalised bbox
        '''
        confidence_scores ,bboxes = self.bins_predictor.predict(frame) 
        highest_confidence_indexes = self.top_n_argmax(confidence_scores, 2)
        bins_bboxes = bboxes[highest_confidence_indexes]
        
        print('🗑️ bins detected,', bins_bboxes)
        
        if debug :
            fig, ax = plt.subplots(1)
            plt.title('bins position prediction')
            ax.imshow(frame)
            
            height, width, _ = frame.shape
            
            for bbox in bins_bboxes:

                ymin, xmin, ymax, xmax  = bbox

                # Convert normalized coordinates to pixel values
                xmin = int(xmin * width)
                ymin = int(ymin * height)
                xmax = int(xmax * width)
                ymax = int(ymax * height)

                # Calculate the box width and height
                box_width = xmax - xmin
                box_height = ymax - ymin

                # Create a Rectangle patch
                rect = plt.Rectangle((xmin, ymin), box_width, box_height, linewidth=3, edgecolor='r', facecolor='none')

                # Add the patch to the axes
                ax.add_patch(rect)
            plt.show()
        
        return bins_bboxes
    

    def predict_goal_position(self, frame, debug):
        '''
        Predict where the goal is in frame, return the normalised bbox
        '''
        confidence_scores ,bboxes = self.goal_predictor.predict(frame) 
        highest_confidence_index = np.argmax(confidence_scores)
        goal_bbox = bboxes[highest_confidence_index]
        
        if debug :
            fig, ax = plt.subplots(1)
            plt.title('goal position prediction')
            ax.imshow(frame)
            
            height, width, _ = frame.shape

            ymin, xmin, ymax, xmax  = goal_bbox

            # Convert normalized coordinates to pixel values
            xmin = int(xmin * width)
            ymin = int(ymin * height)
            xmax = int(xmax * width)
            ymax = int(ymax * height)

            # Calculate the box width and height
            box_width = xmax - xmin
            box_height = ymax - ymin

            # Create a Rectangle patch
            rect = plt.Rectangle((xmin, ymin), box_width, box_height, linewidth=3, edgecolor='r', facecolor='none')
            
            # Add the patch to the axes
            ax.add_patch(rect)
            plt.show()
        
        return goal_bbox
    
    def calculate_accuracy(self, ball_track, targets, goal_pos, bin_positions, action_id, speed_threshold = 0.03, delta_direction_threshold = 2, debug=False, debug_frames=None):
         
        if ball_track is None:
            
            metadata = {'raw_impact_point':None, 
                        'raw_distance': None,
                        'raw_score': 0,
                        'goal_impact_point': None,
                        'goal_position': None,
                        'path_points': None, 
                        'impact_time_seconds': 0 ,
                        'accuracy_plot': None }

            return 0, metadata
            
        
        # get midpoints of boxes
        midpoints = ball_track.midpoints
        
        if debug:
            plt.title('midpoints')
            plt.plot([x for x,y in midpoints], [y for x,y in midpoints])
            plt.scatter([x for x,y in midpoints], [y for x,y in midpoints], c=list(range(len(midpoints))))
            plt.xlim(0,1)
            plt.ylim(1,0)
            plt.show()
        
        # get distances         
        distances = ball_track.distances
        
        if debug:
            plt.title('distances')
            plt.plot(distances)
            plt.show()
        
        # get speeds to use as "kicking threshold"         
        speeds = distances
        # print(speeds)
        
        if debug:
            plt.title('speeds')
            plt.hlines(y=speed_threshold, colors='green', ls='--', lw=2, xmin=0, xmax=len(speeds))
            plt.plot(speeds)
            plt.show()
        
        # index of when kick happened 
        midpoints_over_speed_thresholder = np.where(np.array(speeds) > speed_threshold)
        if len(midpoints_over_speed_thresholder[0])>0:
            
            kick_impulse_index = np.where(np.array(speeds) > speed_threshold)[0][0]
            print('using speed threshold trigger at index ', kick_impulse_index)
        else:
            kick_impulse_index = 0
        print('kick happened at ', kick_impulse_index)
        
        # just the midpoints of the ball after the kick moment         
        post_kick_midpoints = midpoints[kick_impulse_index:]
        
        if debug:
            plt.title('post_kick_midpoints')
            plt.plot([x for x,y in post_kick_midpoints], [y for x,y in post_kick_midpoints])
            plt.scatter([x for x,y in post_kick_midpoints], 
                        [y for x,y in post_kick_midpoints], 
                        c=list(range(len(post_kick_midpoints))))
            plt.xlim(0,1)
            plt.ylim(1,0)
            plt.show()
        
        # get directions after kick
        directions = ball_track.directions[kick_impulse_index:]
        
        if debug:
            plt.title('directions (post kick impulse index)')
            plt.plot(directions, marker='o')
            plt.show()
        
        # get delta directions after kick
        delta_directions = np.array(ball_track.delta_directions[kick_impulse_index:])
        
        if debug:
            plt.title('delta_directions (post kick impulse index)')
            plt.hlines(y=delta_direction_threshold, colors='green', ls='--', lw=2, xmin=0, xmax=len(delta_directions))
            plt.plot(delta_directions, marker='o')
            plt.show()
        
        # get impact index 
        mask_delta_threshold_trggered = delta_directions > delta_direction_threshold
        # if the delta direction thrshold isn't trigger just use the max          
        if mask_delta_threshold_trggered.sum() <= 0:
            if len(delta_directions) <= 0:
                impact_index = 0
            else:
                impact_index = np.argmax(delta_directions) + 2
        else:
            impact_index = np.where(delta_directions > delta_direction_threshold)[0][0] + 2 
        
        # get point of where impact happened
        impact_point = post_kick_midpoints[impact_index]
        print('impact index ', impact_index, 'impact point' , impact_point)
        
        
        closest_distance_to_boxes = self.calculate_closest_distance_to_boxes(impact_point, bin_positions)
        
        print(f'🎯 closest_distance_to_boxes : {closest_distance_to_boxes}')
        
        # noramalise bin location to the space of goal          
        nomalised_bins = [self.normalize_box(b, goal_pos) for b in bin_positions]
        normalised_imapact_on_goal = self.normalize_point([impact_point[0], impact_point[1]], goal_pos)        
        
        normed_closest_distance_to_boxes = self.calculate_closest_distance_to_boxes(normalised_imapact_on_goal, nomalised_bins)
        print(f'📐🎯 normed_closest_distance_to_boxes : {normed_closest_distance_to_boxes}')
        
        denormalised_targets = [self.denormalize_point(t, goal_pos) for t in targets]
        
        
#         import pickle

#         # Create the object to be pickled
#         obj = {'frames': debug_frames,
#                "ball_track":ball_track,
#                "impact_index":impact_index,
#                "goal_pos":goal_pos, 
#                'targets':targets}

#         # Open the file to pickle the object to
#         with open("accracy_plot_data2.pickle", "wb") as f:
#             pickle.dump(obj, f)
        
        
        
        accuracy_plot_file_name = "accuracy_plot.png"
        APSX_Plotter.plot_accuracy_plot(debug_frames, ball_track, impact_index+kick_impulse_index, 
                                        goal_pos, bin_positions, accuracy_plot_file_name)
        # upload to google storage
        asset_bucket = 'apsx-misc-data'
        accuracy_plot_gsl = f'{self.asset_bucket}/{action_id}/{accuracy_plot_file_name}'
        APSX_GCS_utils.upload_to_gcs(accuracy_plot_file_name, accuracy_plot_gsl)

            
        if debug:
            
            # display shot on goal position with targets             
            fig, ax = plt.subplots(1)
            
            plt.title('impact point on goal')
            plt.xlim(-0.2,1.2)
            plt.ylim(1.2,-0.2)
            
            # display impact relative to goal space             

            print(' impact point', impact_point)
            print('norm impact point', normalised_imapact_on_goal)
            plt.plot(normalised_imapact_on_goal[0], normalised_imapact_on_goal[1], 
                     markersize=20, marker='x', markeredgewidth=3)
            
            # display targets relative to goal space             
            plt.scatter([x for x,y in targets],
                     [y for x,y in targets], s=90, marker='X')

            # draw goal             
            ymin, xmin, ymax, xmax  = (0, 0, 1, 1)
            # Calculate the box width and height
            box_width = xmax - xmin
            box_height = ymax - ymin
            # Create a Rectangle patch
            rect = plt.Rectangle((xmin, ymin), box_width, box_height, linewidth=3, edgecolor='r', facecolor='none')
            
            # Add the patch to the axes
            ax.add_patch(rect)
            
            # draw bins   
            for bin_bbox in nomalised_bins:
                ymin, xmin, ymax, xmax  = bin_bbox
                # Calculate the box width and height
                box_width = xmax - xmin
                box_height = ymax - ymin
                # Create a Rectangle patch
                rect = plt.Rectangle((xmin, ymin), box_width, box_height, linewidth=3, edgecolor='r', facecolor='none')

                # Add the patch to the axes
                ax.add_patch(rect)
            
            plt.show()
        
        
        # # calculate score 
        score = self.calculate_score_with_bins(normed_closest_distance_to_boxes)
        # score, closest_target_distance, raw_score = self.calculate_score(normalised_imapact_on_goal, targets)
        
        # get time in seconds when ball impacted         
        impact_time = self.every_nth_frame * impact_index * self.seconds_between_frames
        # get ball path data         
        ball_path = [{'x':m[0], 'y':m[1]} for m in midpoints]
        
        metadata = {'raw_impact_point':impact_point, 
                    'raw_distance': normed_closest_distance_to_boxes,
                    'raw_score': normed_closest_distance_to_boxes,
                    'goal_impact_point': normalised_imapact_on_goal,
                    'goal_position': goal_pos.tolist(),
                    # 'path_points': ball_path, 
                    'impact_time_seconds': impact_time ,
                    'accuracy_plot': accuracy_plot_gsl }
        
        return score, metadata
    
    @staticmethod
    def calculate_score_with_bins(closest_distance_to_bin):

        score = int(Accuracy_Analyser.scale_and_limit_value(1-closest_distance_to_bin, 
                                                            old_min= 0.5, old_max=1, 
                                                            new_min=50, new_max=99))

        return score

    
    @staticmethod
    def calculate_score(point, targets):
    
        # deduct points if a miss is detected     
        miss_penalty = 0
        if not ((0 < point[0] < 1) and  (0< point[1])):
            miss_penalty = 40
            # return 0

        distances_to_targets = Accuracy_Analyser.get_distances_to_targets(point, targets)
        closest_target_distance = min(distances_to_targets)
        raw_score = 1 - closest_target_distance

        score = int(Accuracy_Analyser.scale_and_limit_value(raw_score, old_min= 0.3, old_max=0.68, new_min=50, new_max=99))

        return score - miss_penalty, closest_target_distance, raw_score
    
    @staticmethod
    def calculate_closest_distance_to_boxes(point, boxes):
        
        distances = []
        for b in boxes:
            dist = Accuracy_Analyser.distance_to_box(point, b)
            distances.append(dist)
            
        return min(distances)
    
    @staticmethod
    def distance_to_box(point, box):
        ymin, xmin, ymax, xmax = box
        x, y = point

        # Check if the point is inside the box
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return 0

        # Calculate the distances to the edges of the box
        if ymin <= y <= ymax:
            y_distance = 0
        else:
            y_distance = min(abs(y - ymin), abs(y - ymax))


        if  xmin <= x <= xmax:
            x_distance = 0
        else:
            x_distance = min(abs(x - xmin), abs(x - xmax))

        distance = math.sqrt(x_distance**2 + y_distance**2)
    
        print(distance)
        # Return the minimum distance
        return distance
    
    @staticmethod
    def get_distances_to_targets(point, targets):

        x ,y = point
        distances = []

        for target_x, target_y in targets:
            print(target_x, target_y )
            dist = math.sqrt( (x - target_x)**2 + (y - target_y)**2 )
            print(dist)
            distances.append(dist)

        return distances
    
    @staticmethod
    def scale_and_limit_value(value, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        scaled_value = (((value - old_min) * new_range) / old_range) + new_min
        limited = min(max(scaled_value, new_min), new_max)
        return limited
    
    @staticmethod
    def normalize_box(box_to_normalise, base_box):
        ymin, xmin, ymax, xmax = box_to_normalise
        
        normalized_xmin, normalized_ymin = Accuracy_Analyser.normalize_point((xmin, ymin), base_box)
        normalized_xmax, normalized_ymax = Accuracy_Analyser.normalize_point((xmax, ymax), base_box)

        return (normalized_ymin, normalized_xmin, normalized_ymax, normalized_xmax)
    
    @staticmethod
    def normalize_point(point, box):
        ymin, xmin, ymax, xmax = box
        x, y = point

        normalized_x = (x - xmin) / (xmax - xmin)
        normalized_y = (y - ymin) / (ymax - ymin)

        return normalized_x, normalized_y
    

    @staticmethod
    def denormalize_point(normalized_point, box):
        ymin, xmin, ymax, xmax = box
        normalized_x, normalized_y = normalized_point

        x = normalized_x * (xmax - xmin) + xmin
        y = normalized_y * (ymax - ymin) + ymin

        return x, y
    
