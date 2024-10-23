
import numpy as np
import math

class ObjectTrack:
    def __init__(self, track_id, max_distance, max_missing_frames):
        self.track_id = track_id
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.frames = []
        self.distances = []
        self.midpoints = []
        self.directions = []
        self.delta_directions = []
        self.interpolate_missing_frames=True
        self.interpolated_frame_count = 0
        self.current_missed_frames_count = 0


    @property
    def bboxes(self):
      output = []
      for f in self.frames:
        output.append(f[1].tolist())
      return output


    def add_frame(self, frame_num, bbox):

        if bbox is None:
          self.current_missed_frames_count += 1
        else:
          self.current_missed_frames_count = 0

        self.frames.append((frame_num, bbox))


    def get_missed_frames_count(self):

        return self.current_missed_frames_count

        # missed_frames_count = 0
        # for _, bbox in reversed(self.frames):
        #     if bbox is None:
        #         missed_frames_count+=1
        #     else:
        #         return missed_frames_count

    def interpolate_missing_boxes(self):

        boxes = []
        for _, bbox in self.frames:
            if bbox is not None:
                boxes.append(bbox)
            else:
                # create bbox with Nones for each dimention
                boxes.append([None,None,None,None])
                self.interpolated_frame_count += 1

        interpolated = self.interpolate_nones_matrix(boxes)

        for i, new_data in enumerate(interpolated):
            self.frames[i] = (self.frames[i][0], new_data)


    def end_tracking(self):
        '''
        Remove trailing Nones in track and do post processing
        '''
        len_frames = len(self.frames)
        for i in range(len_frames):

            if self.frames[-1][1] is None:
                self.frames.pop()
            else:
                break

        if self.interpolate_missing_frames:
            self.interpolate_missing_boxes()


        self.distances = self.calculate_all_distances(self.frames)

        self.midpoints =  self.calculate_midpoints(self.frames)

        self.directions = self.get_path_directions(self.midpoints)

        self.delta_directions = self.get_delta_directions(self.directions)


    def pick_best_bbox(self, bbox_list):
        '''
        Pick the most similare in shape box within range.
        '''

        bboxes_in_range = []

        for bbox in bbox_list:
            if self.is_in_range(bbox):
                bboxes_in_range.append((bbox, self.get_shape_similarity_score(bbox)))

        if len(bboxes_in_range):
            return sorted(bboxes_in_range, key=lambda x: x[1])[0][0]
        return None


    def get_last_detected_bbox(self):
        for _ , frame in reversed(self.frames):
            if frame is not None:
                return frame


    def get_shape_similarity_score(self, bbox):
        latest_shape = self.get_latest_shape()
        bbox_shape = self.calculate_shape(bbox)
        score = abs(latest_shape[0] - bbox_shape[0]) + abs(latest_shape[1] - bbox_shape[1])
        return score

    @staticmethod
    def calculate_shape(bbox):
        y_min, x_min, y_max, x_max  = bbox
        width = x_max - x_min
        height = y_max - y_min
        return (width, height)


    def get_latest_shape(self):
        return self.calculate_shape(self.get_last_detected_bbox())


    def is_same_track(self, bbox):

        if not self.frames:
            return False

        _, last_bbox = self.get_last_detected_position()
        distance = self.calculate_distance(bbox, last_bbox)
        return distance <= self.max_distance


    def is_in_range(self, bbox):

        # if there are no frames
        if not self.frames:
            return False

        last_bbox = self.get_last_detected_bbox()
        distance = self.calculate_distance(bbox, last_bbox)

        # increase distance to compensate for missed frames
        # TODO this is abit simple, could be looking at extrapolated distance
        distance_boundary = self.max_distance + (self.current_missed_frames_count *  self.max_distance)

        return distance <= distance_boundary


    @staticmethod
    def calculate_all_distances(frames):

        distances = []
        for i in range(len(frames)-1):
            distances.append( ObjectTrack.calculate_distance(frames[i][1],
                                                             frames[i+1][1]) )
        return distances

    @staticmethod
    def calculate_distance(bbox1, bbox2):
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        center1 = np.array([(y1_min + y1_max) / 2, (x1_min + x1_max) / 2])
        center2 = np.array([(y2_min + y2_max) / 2, (x2_min + x2_max) / 2])
        distance = np.linalg.norm(center1 - center2)
        return distance


    @staticmethod
    def calculate_midpoints(bbox_list):
        midpoints = []
        for seconds, bbox in bbox_list:
            ymin, xmin, ymax, xmax = bbox
            midpoint_x = (xmin + xmax) / 2
            midpoint_y = (ymin + ymax) / 2
            midpoint = (midpoint_x, midpoint_y)
            midpoints.append(midpoint)
        return midpoints


    @staticmethod
    def interpolate_nones_matrix(matrix):
        # Convert the matrix to a NumPy array
        arr = np.array(matrix, dtype=float)

        # Iterate through each column
        for col in range(arr.shape[1]):
            # Find indices of None values in the current column
            none_indices = np.where(np.isnan(arr[:, col]))[0]

            # Find indices of non-None values in the current column
            non_none_indices = np.where(~np.isnan(arr[:, col]))[0]

            # do linear interpolation
            arr[none_indices, col] = np.interp(none_indices, non_none_indices, arr[non_none_indices, col])
        return arr


    @staticmethod
    def get_path_directions(path):

        x = np.array([point[0] for point in path])
        y = np.array([point[1] for point in path])

        delta_x = x[1:] - x[:-1]
        delta_y = y[1:] - y[:-1]

        directions = np.arctan2(delta_y, delta_x)

        return directions


    @staticmethod
    def get_delta_directions(directions):
        delta_directions = []
        for i in range(len(directions)-1):
          delta = ObjectTrack.calculate_radian_difference(directions[i], directions[i+1])
          delta_directions.append(delta)
        # delta_direction = np.abs(directions[1:]-directions[:-1])
        return delta_directions


    @staticmethod
    def calculate_radian_difference(direction1, direction2):
        direction1 = direction1 % (2 * math.pi)
        direction2 = direction2 % (2 * math.pi)
        absolute_difference = abs(direction1 - direction2)
        angular_difference = min(absolute_difference, 2 * math.pi - absolute_difference)

        return angular_difference


    def max_speed(self):
        return max(self.distances)


class ObjectTracker:
    def __init__(self, max_distance=1., max_missing_frames=0):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.tracks = []
        self.next_track_id = 1


    def get_longest_track(self):
        sorted_tracks = sorted(self.tracks, key=lambda track: len(track.frames))
        if len(sorted_tracks):
          return sorted_tracks[-1]
        else:
          return None


    @staticmethod
    def copy_bbox_array(bbox_array):

      # handle numpy list different from primative
      if type(bbox_array) in {tuple, list}:
        # primative
        return bbox_array[:]
      else:
        # numpy
        return bbox_array.tolist().copy()


    def analyse(self, bbox_lists):

        # print(type(bbox_lists), type(bbox_lists[0]), type(bbox_lists[0][0]))

        # assert(type(bbox_lists) == list)
        # assert(type(bbox_lists[0]) in [list, tuple])
        # assert(type(bbox_lists[0][0]) in [list, tuple])
        # assert(type(bbox_lists[0][0][0]) in [int, float])

        print(f'Running APSX object tracker on {len(bbox_lists)} frames')

        active_tracks = []

        for frame_num, bboxes in enumerate(bbox_lists):

            frame_bboxes = self.copy_bbox_array(bboxes)

            # longest track get first dibs TODO future do match making
            sorted_tracks = sorted(active_tracks,
                                   key=lambda obj: len(obj.frames),
                                   reverse=True)

            for track in sorted_tracks:
                # for each obeject track, find best match bbox in frame
                best_box = track.pick_best_bbox(frame_bboxes)
                # add box even if its None
                track.add_frame(frame_num, best_box)

                if best_box is not None:
                    # print('best box ', best_box)
                    frame_bboxes.remove(best_box)
                else:
                    if track.get_missed_frames_count() > self.max_missing_frames:
                        track.end_tracking()
                        active_tracks.remove(track)

            # start tracking any remaining boxes
            for remaining_bbox in frame_bboxes:
                new_track = ObjectTrack(self.next_track_id, self.max_distance, self.max_missing_frames)
                new_track.add_frame(frame_num, remaining_bbox)
                self.next_track_id += 1
                self.tracks.append(new_track)
                active_tracks.append(new_track)

        # finish tracking all objects when all frames analysed
        for active_track in active_tracks:
            active_track.end_tracking()

        for a_t in active_tracks:
            print(a_t.frames[0])


    def get_tracks(self):
        return self.tracks
