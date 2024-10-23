
##require opencv-python
##require numpy
##require tensorflow
##require scipy

import tempfile

import cv2
import numpy as np
from scipy.ndimage import label
import tensorflow as tf


class PlayerSegmentation:
    
    def __init__(self, model_path="pose_landmark_heavy.tflite"):
        """
        Loads the TensorFlow Lite model for player segmentation.

        Args:
            model_path (str): Path to the TensorFlow Lite model file.
        """

        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        
    @staticmethod 
    def load_image(image_path):
        # Read the image in color mode
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    def preprocess_image(self, image):
        """
        resizes, and preprocesses it for the model.

        Args:
            image: image matrix.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """

        # Get model input shape and data type
        model_input_shape = tuple(self.input_details[0]["shape"][1:3])
        model_input_dtype = self.input_details[0]["dtype"]

        # Resize the image
        image = cv2.resize(image, dsize=model_input_shape)

        # Convert to model input format
        image = image.astype(model_input_dtype)

        # Normalize pixel values (optional, adjust based on your model)
        image = image / 255.0  # Example: Normalize to range [0, 1]

        return np.expand_dims(image, axis=0)
    
    
    def get_image_segment_bytes(self, image_path):
        
        mask = self.segmenter.segment_image(image_path)
        image = self.boolean_matrix_to_image(mask)
        image_bytes = self.numpy_image_to_bytes(image)
        return image_bytes
    
    
    def run_segmentation(self, input_data):
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run model inference
        self.interpreter.invoke()

        # Get the output mask
        output_data = self.interpreter.get_tensor(self.output_details[2]["index"])
        
        return output_data[0]
    

    def run_pose(self, input_data):
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run model inference
        self.interpreter.invoke()

        # Get the output mask
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        return output_data

    
    def segment_image_to_file(self, image_path, threshold=5):
        
        mask_np = self.segment_image(image_path, threshold)  
        
        person_location = self.biggest_person_location(mask_np)
        
        mask_image_np = self.boolean_matrix_to_image(mask_np, invert=False)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name + '.jpg'
            
        self.save_numpy_image(mask_image_np, filename)
        
        return {'filename':filename, 'person_location':person_location}
        

    
    def segment_image(self, image_path, threshold=5):
        """
        Segments an image using the loaded model and resizes the mask.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Segmented image mask as a NumPy array.
        """
        numpy_image = self.load_image(image_path)        
        mask = self.segment_numpy_image(numpy_image, threshold=5)
        return mask
    
    

    def crop_numpy_to_person(self, numpy_image, padding_distance=0.1):
        
        mask = self.segment_numpy_image(numpy_image, threshold=5)
        person_location = self.biggest_person_location(mask)
        
        if person_location:
            cropped_numpy = self.crop_person(numpy_image, 
                                        person_location, 
                                        padding_distance=padding_distance, 
                                        fill_value=255)
            return cropped_numpy
        
        print('no crop data , returning orginal image')
        return numpy_image
        
        

    
    
    def segment_numpy_image(self, numpy_image, threshold=5):
        
        preprocessed_image = self.preprocess_image(numpy_image)
        mask = self.run_segmentation(preprocessed_image)

        # Resize the mask to match the original image size
        original_height, original_width = numpy_image.shape[:2]
        resized_mask = cv2.resize(mask, dsize=(original_width, original_height))
        
        if threshold is None:
            return resized_mask
        
        binarized_mask = resized_mask > threshold
        
        return binarized_mask
    
    
    @staticmethod
    def boolean_matrix_to_image(boolean_matrix, invert=False):
        """
        Converts a boolean matrix to a black and white image file.

        Args:
          boolean_matrix (np.ndarray): A 2D boolean numpy array representing the image.
              True values are considered white, False values are black.
          filename (str, optional): The filename to save the image to. Defaults to "output_image.png".
        """
        # Invert the matrix as OpenCV uses 0 for black and 255 for white
        if invert:
            image = np.where(boolean_matrix, 255, 0).astype(np.uint8)
        else:
            image = np.where(boolean_matrix, 0, 255).astype(np.uint8)

        return image
        
    @staticmethod
    def save_numpy_image(image, filename):
        """
        Saves a NumPy image to a local file.

        Args:
          image (np.ndarray): A NumPy array representing the image.
          filename (str): The filename to save the image under.
        """
        # Write the image to disk
        cv2.imwrite(filename, image)

        
    @staticmethod
    def numpy_image_to_bytes(image):
        """
        Converts a NumPy image to a byte array representing the image data.

        Args:
          image (np.ndarray): A NumPy array representing the image.

        Returns:
          bytes: The image data as a byte array.
        """
        # Encode the image as PNG (or adjust the format as needed)
        retval, buffer = cv2.imencode('.png', image)

        # Check if encoding was successful
        if retval:
            return buffer.tobytes()
        else:
            print("Error converting image to bytes!")
            return None
        

    @staticmethod
    def apply_mask(base_image_path, mask_image_path):
        """
        Loads two images, applies a mask to the base image.

        Args:
          base_image_path (str): Path to the base image file.
          mask_image_path (str): Path to the black and white mask image file.
        """

        # Load the base image and convert to BGR format (OpenCV standard)
        base_image = cv2.imread(base_image_path)
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)

        # Load the mask image and ensure it's grayscale
        mask = np.expand_dims(cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE), axis=-1)

        # Invert the mask (assuming white in mask represents keep)
        inverted_mask = np.where(mask == 0, 0, 1)

        # Apply bitwise AND operation to mask out the base image
        masked_image = base_image * inverted_mask
        
        return masked_image
    
    
    @staticmethod
    def biggest_person_location(mask):
        
        blobs = PlayerSegmentation.find_blobs(mask)
        
        if blobs:
            return blobs[0]
        
        return None
    
    @staticmethod
    def find_blobs(image):
        """
        Finds islands in a binary image and returns a list of dictionaries containing
        island properties.

        Args:
        image: A 2D numpy array representing the binary image.

        Returns:
        A list of dictionaries. Each dictionary contains the following keys:
          label: The label value of the island.
          size: The total number of pixels in the island.
          centroid: A tuple containing the average (x, y) coordinates of the island.
          bounding_box: A dictionary with keys 'min_x', 'max_x', 'min_y', and 'max_y'
              representing the bounding box coordinates of the island.
          normalized_centroid: A tuple containing the normalized average (x, y) coordinates
              of the island (between 0 and 1).
          normalized_bounding_box: A dictionary with the same keys as 'bounding_box' but
              containing normalized values between 0 and 1. 
        """
        height, width = image.shape[:2]

        # Apply the label function to find connected components
        labeled_matrix, num_labels = label(image)

        # Get the unique labels (excluding background label 0)
        unique_labels = np.unique(labeled_matrix)

        islands = []
        
        for label_value in unique_labels[1:]:
            # Create a mask for the current label
            mask = (labeled_matrix == label_value)

            mask_size = mask.sum()

            # Find the indices of the ones in the mask
            indices = np.where(mask)

            # Calculate island properties
            centroid = (indices[0].mean(), indices[1].mean())
            min_x, max_x = indices[1].min(), indices[1].max()
            min_y, max_y = indices[0].min(), indices[0].max()

            # Normalize coordinates
            normalized_centroid = (centroid[0] / (width - 1), centroid[1] / (height - 1))
            normalized_bounding_box = {
                'min_x': min_x / (width - 1),
                'max_x': max_x / (width - 1),
                'min_y': min_y / (height - 1),
                'max_y': max_y / (height - 1)
            }

            # Create island data dictionary
            island_data = {
                "label": label_value,
                "size": mask_size,
                "centroid": centroid,
                "bounding_box": {
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                },
                "normalized_centroid": normalized_centroid,
                "normalized_bounding_box": normalized_bounding_box
            }

            islands.append(island_data)
            
        sorted_islands_by_size = sorted(islands, key=lambda island: island["size"], reverse=True) 


        return sorted_islands_by_size

    @staticmethod
    def visualize_blobs(image, islands):
        """
        Visualizes the identified islands on the provided image.

        Args:
          image: A 2D numpy array representing the binary image.
          islands: A list of dictionaries containing island properties as returned by
              the `find_islands` function.
          debug: (Optional) A boolean flag to print additional information.
        """

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.gca().set_aspect('equal')

        for island in islands:
            centroid = island["centroid"]
            bounding_box = island["bounding_box"]

            circle = plt.Circle(centroid, radius=2, color='red', fill=True)
            rect = plt.Rectangle((bounding_box['min_x'], bounding_box['min_y']),
                                  bounding_box['max_x'] - bounding_box['min_x'],
                                  bounding_box['max_y'] - bounding_box['min_y'],
                                  linewidth=1, edgecolor='r', facecolor='none')

            plt.gca().add_patch(circle)
            plt.gca().add_patch(rect)

            

        
        
            
    @staticmethod
    def crop_person(image, person_data, padding_distance=0.1, fill_value=255):
        """
        Crops a square region of the person from the given image based on the person information.
        If the person is close to the edge, remaining overlapped edges are filled with white pixels.
        The person is always centered within the cropped square.

        Args:
          image: A 2D numpy array representing the image.
          person_data: A dictionary containing person properties, including bounding box information.
          padding_distance: (Optional) A float value between 0 and 1 specifying the padding amount
              as a normalized distance of the image width. Defaults to 0.1.
          fill_value: (Optional) The value to use for filling the padded edges. Defaults to 255 (white).

        Returns:
          A new numpy array containing the cropped person region with white pixel padding,
          or None if the person is entirely outside the image boundaries.
        """

        # Extract bounding box information
        person_bbox = person_data['bounding_box']

        # Calculate padding based on normalized value and image width
        image_width, image_height = image.shape[1], image.shape[0]
        padding_px = int(image_width * padding_distance)

        # Expand bounding box with padding
        expanded_bbox = {
          'min_x': max(0, person_bbox['min_x'] - padding_px),
          'max_x': min(image_width, person_bbox['max_x'] + padding_px),
          'min_y': max(0, person_bbox['min_y'] - padding_px),
          'max_y': min(image_height, person_bbox['max_y'] + padding_px)
        }

        # Check if person is entirely outside the image after padding
        if (expanded_bbox['min_x'] < 0 or expanded_bbox['max_x'] > image_width or
          expanded_bbox['min_y'] < 0 or expanded_bbox['max_y'] > image_height):
            return None

        # Determine square dimensions based on expanded bounding box
        square_width = max(expanded_bbox['max_x'] - expanded_bbox['min_x'], expanded_bbox['max_y'] - expanded_bbox['min_y'])

        print('square_width', square_width)
        
        # Calculate offsets to center the square within the valid area
        # Consider both person centroid and potential padding
        person_center_x = (expanded_bbox['min_x'] + expanded_bbox['max_x']) / 2
        person_center_y = (expanded_bbox['min_y'] + expanded_bbox['max_y']) / 2
        offset_x = int(max(0, person_center_x - square_width / 2) - expanded_bbox['min_x'])
        offset_y = int(max(0, person_center_y - square_width / 2) - expanded_bbox['min_y'])

        # Limit offsets to not exceed image boundaries
        offset_x = min(offset_x, image_width - square_width - (expanded_bbox['max_x'] - image_width))
        offset_y = min(offset_y, image_height - square_width - (expanded_bbox['max_y'] - image_height))

        # Adjust start coordinates for square crop with padding
        start_x = max(expanded_bbox['min_x'] + offset_x, 0)
        start_y = max(expanded_bbox['min_y'] + offset_y, 0)

        # Handle single-channel and RGB images differently for padding
        if len(image.shape) == 2:  # Single-channel image
            # Prepare the cropped region with padding (filled with fill_value)
            cropped_person = np.full((square_width, square_width), fill_value, dtype=image.dtype)
        else:  # RGB image
            # Prepare the cropped region with padding (filled with black pixels for RGB)
            cropped_person = np.full((square_width, square_width, 3), fill_value=[fill_value, fill_value, fill_value], dtype=image.dtype)

        # Copy the overlapping image content into the cropped region
        end_x = min(start_x + square_width, image_width)
        end_y = min(start_y + square_width, image_height)

        # print(cropped_person.shape, end_y , start_y, end_x , start_x )
        cropped_person[:end_y - start_y, :end_x - start_x] = image[start_y:end_y, start_x:end_x]
            

        # Handle single-channel and RGB images differently for padding
        if len(image.shape) == 2:  # Single-channel image
            # Prepare the cropped region with padding (filled with fill_value)
            cropped_person[:end_y - start_y, :end_x - start_x] = image[start_y:end_y, start_x:end_x]
        else:  # RGB image
            # Prepare the cropped region with padding (filled with black pixels for RGB)
            cropped_person[:end_y - start_y, :end_x - start_x,:] = image[start_y:end_y, start_x:end_x, :]


        return cropped_person
