
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["figure.dpi"] = 200

GOOGLE_BLUE = '#4285F4'
GOOGLE_RED  = '#DB4437'
GOOGLE_YELLOW ='#F4B400'
GOOGLE_GREEN ='#0F9D58'


# Create a list of colors for the color map
colors = [GOOGLE_BLUE, GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN]

# Create the color map
cmap = LinearSegmentedColormap.from_list('Google', colors)

if not plt.colormaps.get('Google'):
    plt.colormaps.register(cmap)
    
# Create a list of colors for the color map
colors2 = [GOOGLE_BLUE, "#FFFFFF"]

# Create the color map
cmap2 = LinearSegmentedColormap.from_list('Google_blue', colors2)

if not plt.colormaps.get('Google_blue'):
    plt.colormaps.register(cmap2)


landmarks_of_intrest_set = {'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
                         'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'}

bone_pairs = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
              ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
              ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
              ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
              ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
              ("right_hip", "right_knee"), ("right_knee", "right_ankle")
             ]


def extract_bone_segments(bone_pairs, landmark_map):

    bone_segments = []

    for (bone_1, bone_2) in bone_pairs:
        # if both landmarks exists     
        if bone_1 in landmark_map and bone_2 in landmark_map:
            segment = (landmark_map[bone_1], landmark_map[bone_2])
            bone_segments.append(segment)
            
    return bone_segments


def extract_landmark_data(track_frame):
    
    landmarks = track_frame.landmarks
    
    landmark_xs = []
    landmark_ys = []
    landmark_map = {}
    
    for l in landmarks:

        if l.name not in  landmarks_of_intrest_set:
            continue

        x, y = l.point.x, l.point.y
        landmark_xs.append(x)
        landmark_ys.append(y)
        landmark_map[l.name] = (x, y)
        
    return landmark_xs, landmark_ys, landmark_map


def extract_rect_bbox(track_frame):
    
    bbox = track_frame.normalized_bounding_box

    rect_x = bbox.left 
    rect_y = bbox.top 
    rect_width = (bbox.right - bbox.left) 
    rect_height = (bbox.bottom - bbox.top) 
    
    return (rect_x, rect_y, rect_width, rect_height)


def plot_pose(frame, rect, landmark_xs, landmark_ys, bone_segments, file_name=None):
    
    
    
    (rect_x, rect_y, rect_width, rect_height) = rect
    
    plt.figure().clear()

    # Create a figure and axes
    fig, ax = plt.subplots()
    
    plt.margins(x=0, y=0)
    ax.margins(x=0, y=0)

    # Turn off the axes
    ax.axis('off')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Add the image to the plot
    ax.imshow(frame, extent=[0, 1, 1, 0], aspect=1.5)

    # Create a rectangle object
    rectangle = mpatches.Rectangle(xy=(rect_x, rect_y), width=rect_width, height=rect_height, fill=False, 
                                   color=GOOGLE_RED, linewidth=2, linestyle='--')

    line_collection = mcollections.LineCollection(bone_segments)
    line_collection.set_color(GOOGLE_BLUE)
    line_collection.set_linewidth(2)
    line_collection.set_linestyle("dashed")
    ax.add_collection(line_collection)

    ax.scatter(landmark_xs, landmark_ys, s = 10, c=GOOGLE_YELLOW, zorder=2)

    # Add the rectangle to the plot
    ax.add_patch(rectangle)

    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 )
    else:
        plt.show()
        

def bbox_to_rect(bbox):
    
    y_min, x_min, y_max, x_max  = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    return (x_min, y_min, width, height)

def coords_to_xy(coords):
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    
    return xs, ys
        
        
def plot_ball_bboxes(image_frames, ball_track, kick_index, file_name=None):
    
    # Create a figure and axes
    plt.figure().clear()
    fig, ax = plt.subplots()
    
    plt.margins(x=0, y=0)
    ax.margins(x=0, y=0)

    # Turn off the axes
    ax.axis('off')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    
    ax.imshow(image_frames[kick_index], extent=[0, 1, 1, 0], aspect=1.5)
    
    bboxes = ball_track.frames[kick_index:]
    
    xs, ys = coords_to_xy(ball_track.midpoints[kick_index:])
    
    ax.plot(xs, ys, c=GOOGLE_RED, zorder=0, linestyle='--', linewidth=2)
    
    for _, bbox in bboxes:
        
        (rect_x, rect_y, rect_width, rect_height) = bbox_to_rect(bbox)
            # Create a rectangle object
        rectangle = mpatches.Rectangle(xy=(rect_x, rect_y), width=rect_width, height=rect_height, fill=False, 
                                   color=GOOGLE_BLUE, linewidth=2, linestyle='--', hatch='xxxx')

        ax.add_patch(rectangle)
        
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 )
    else:
        plt.show()
        
    

        
def plot_frame_pos(frame_image, frame_pos_data,  file_name=None):
    
    rect = extract_rect_bbox(frame_pos_data)
    landmark_xs, landmark_ys, landmark_map = extract_landmark_data(frame_pos_data)
    bone_segments = extract_bone_segments(bone_pairs, landmark_map)
    
    plot_pose(frame_image, rect, landmark_xs, landmark_ys, bone_segments,  file_name)
    
    

from matplotlib.colors import LinearSegmentedColormap


def plot_boxes_on_frame(frame, boxes):
    
    plt.figure().clear()
    fig, ax = plt.subplots()
    
    plt.margins(x=0, y=0)
    ax.margins(x=0, y=0)

    # Turn off the axes
    ax.axis('off')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    
    plt.imshow(frame, extent=[0, 1, 1, 0], aspect='auto', zorder=-10)
    
    
    for box in boxes:
#         vertex ai format
        # xmin, xmax, ymin, ymax  = box 
#     tflite
        ymin, xmin, ymax, xmax  = box
        # Calculate the box width and height
        box_width = xmax - xmin
        box_height = ymax - ymin
        # Create a Rectangle patch
        rect = plt.Rectangle((xmin, ymin), box_width, box_height, color=GOOGLE_RED, 
                             linewidth=2, fill=False, zorder=-5)

        ax.add_patch(rect)
        
    plt.show()
    

    
def crop_frame_to_pos(frame, pos, padding=0.1, aspect=1, file_name=None):
    
    assert padding <= 0.5, "padding cannot be more than 0.5"
    assert padding >= 0, "padding cannot be less than 0"
    assert aspect <= 5, "aspect cannot be more than 5"
    assert aspect >= 0.1, "aspect cannot be less than 0.1"
    
    
    bbox = pos.normalized_bounding_box
    bbox_top = bbox.top + 0.5 
    bbox_bottom = bbox.bottom + 0.5 
    bbox_left = bbox.left + 0.5 
    bbox_right = bbox.right + 0.5 
        
    height, width, _ = frame.shape
    
    padded_frame = np.ones((height*2, width*2, 3), dtype=np.uint8) * 255
    
    padded_height_start = height//2
    padded_width_start = width//2
    padded_frame[padded_height_start:padded_height_start+height, padded_width_start: padded_width_start+ width, :] = frame
    

    width_mid_point = (bbox_left + bbox_right)/2
    width_mid_point_pixels = width_mid_point * width
        
    pos_height = bbox_bottom - bbox_top
    
    padded_height_pixels = (pos_height + (padding * 2)) * height
    padded_top_pixels = (bbox_top - padding) * height
    padded_bottom_pixels = (bbox_bottom + padding) * height
        
    apected_radius_pixels = (padded_height_pixels * aspect) / 2 
    
    y_min = int( padded_top_pixels)
    y_max = int( padded_bottom_pixels)
    x_min = int( width_mid_point_pixels - apected_radius_pixels)
    x_max = int( width_mid_point_pixels + apected_radius_pixels)
        
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(padded_frame.shape[0], y_max)
    x_max = min(padded_frame.shape[1], x_max)
    
    crop = padded_frame[y_min:y_max, x_min:x_max, :]
    
    plt.figure().clear()
    
    plt.margins(x=0, y=0)
    plt.axis('off')

    plt.imshow(crop)
    
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 )
    else:
        plt.show()

    
def plot_accuracy_plot(frames, ball_track, impact_index, goal_pos, bins_pos, file_name=None):
    
    midpoints = ball_track.midpoints
    
    impact_point = midpoints[impact_index]
    
    plt.figure().clear()
        
    fig, ax = plt.subplots()
    
    plt.margins(x=0, y=0)
    ax.margins(x=0, y=0)

    # Turn off the axes
    ax.axis('off')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # ball path line    
    plt.plot([x for x,y in midpoints[:impact_index]], 
             [y for x,y in midpoints[:impact_index]], 
             c=GOOGLE_YELLOW,zorder=-3, alpha=0.5)

    
    # ball path circles BEFORE impact     
    plt.scatter([x for x,y in midpoints[:impact_index]], 
                [y for x,y in midpoints[:impact_index]],
                c=list(range(impact_index)), s=150, alpha=0.7, linewidths=2, zorder=-2, cmap='Google_blue')
    
                # edgecolors=list(range(len(midpoints))),
               #  edgecolors = GOOGLE_BLUE, facecolor=None, c=None,
               # s= 100, zorder=-2,  linewidths=1 )
    
    # impact marker     
    plt.plot(impact_point[0], impact_point[1], markersize=30, 
             marker='x', markeredgewidth=3, c=GOOGLE_BLUE, zorder= -3)
   
    plt.imshow(frames[int((len(frames) - 1)/2)], 
               extent=[0, 1, 1, 0], aspect='auto', zorder=-10)

    # draw goal             
    ymin, xmin, ymax, xmax  = goal_pos
    # Calculate the box width and height
    box_width = xmax - xmin
    box_height = ymax - ymin
    # Create a Rectangle patch
    rect = plt.Rectangle((xmin, ymin), box_width, box_height, color=GOOGLE_RED, 
                         linewidth=2, linestyle='--', fill=False, zorder=-5)
    
    ax.add_patch(rect)
    
    # draw bins     
    for bin_bbox in bins_pos:
        ymin, xmin, ymax, xmax  = bin_bbox
        # Calculate the box width and height
        box_width = xmax - xmin
        box_height = ymax - ymin
        # Create a Rectangle patch
        rect = plt.Rectangle((xmin, ymin), box_width, box_height, color=GOOGLE_RED, 
                             linewidth=2, linestyle='--', fill=False, zorder=-5)

        ax.add_patch(rect)

    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 )
    else:
        plt.show()
        
        

def plot_impacts(impacts, file_name=None):

    # impacts = [[0.5, 0.5], [0.2, 0.8], [0.1, 0.1], [-0.1, 0.9]]

    plt.figure().clear()

    plt.gca().axis('off')

    plt.gca().set_aspect('equal')

    rect = plt.Polygon([[0,1],[0,0],[2,0],[2,1]], closed=False,  facecolor = 'none',
                         edgecolor='black',
                        linewidth=10)

    # Add the patch object to the axes
    plt.gca().add_patch(rect)
    
    bin_width = 0.3
    
    top_bin1 = plt.Rectangle([0.1,0.1],bin_width,bin_width, facecolor = 'none',
                         edgecolor=GOOGLE_YELLOW,
                        linewidth=15)
    plt.gca().add_patch(top_bin1)
    
    top_bin2 = plt.Rectangle([1.9 - bin_width,0.1],bin_width,bin_width, facecolor = 'none',
                         edgecolor=GOOGLE_YELLOW,
                        linewidth=15)
    plt.gca().add_patch(top_bin2)


    for ball in impacts:
        ball_x = min(max(ball[0], -0.05), 1.05)    
        ball_y = min(max(ball[1], -0.1), 1.1) 
        circle = plt.Circle((ball_x*2, ball_y), radius=0.1, 
                            facecolor='white', edgecolor='black', linewidth=5)

        # Add the patch object to the axes
        plt.gca().add_patch(circle)


    # Set the limits of the plot
    plt.xlim(-0.5, 2.5)
    plt.ylim( 1.25, -0.25)


    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 ,transparent=True)
    else:
        plt.show()
        
        
def plot_impacts_for_card(impacts, file_name=None):

    # impacts = [[0.5, 0.5], [0.2, 0.8], [0.1, 0.1], [-0.1, 0.9]]

    plt.figure().clear()

    plt.gca().axis('off')

    plt.gca().set_aspect('equal')

    rect = plt.Polygon([[0,1],[0,0],[2,0],[2,1]], closed=False,  facecolor = 'none',
                         edgecolor='white',
                        linewidth=15)

    # Add the patch object to the axes
    plt.gca().add_patch(rect)
    
    bin_width = 0.3
    
    top_bin1 = plt.Rectangle([0,0],bin_width,bin_width, facecolor = 'none',
                         edgecolor='white',
                        linewidth=15)
    plt.gca().add_patch(top_bin1)
    
    top_bin2 = plt.Rectangle([2 - bin_width,0],bin_width,bin_width, facecolor = 'none',
                         edgecolor='white',
                        linewidth=15)
    plt.gca().add_patch(top_bin2)


    for ball in impacts:
        ball_x = min(max(ball[0], -0.05), 1.05)    
        ball_y = min(max(ball[1], -0.1), 1.1)    
        
        circle = plt.Circle((ball_x*2, ball_y), radius=0.15, 
                            facecolor=GOOGLE_BLUE, edgecolor='white', linewidth=12)

        # Add the patch object to the axes
        plt.gca().add_patch(circle)


    # Set the limits of the plot
    plt.xlim(-0.25, 2.25)
    plt.ylim( 1.25, -0.25)


    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0 ,transparent=True)
    else:
        plt.show()


