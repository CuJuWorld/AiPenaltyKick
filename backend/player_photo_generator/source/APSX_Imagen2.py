
##require google-cloud-aiplatform
##require pillow

import tempfile
import base64
from io import BytesIO

from PIL import Image as PIL_Image

import APSX_Logger
import APSX_Player_Segmentation

from vertexai.preview.vision_models import ImageGenerationModel, Image

class Imagen2:
    
    def __init__(self, model_name="imagegeneration@006"):

        self.imagen_model = ImageGenerationModel.from_pretrained(model_name)
        self.segmentation_util = APSX_Player_Segmentation.PlayerSegmentation()
        print('✅ Loaded ', self)
        
#         
    @staticmethod    
    def b64_to_file(b64_image):
        
        decoded_data = base64.b64decode(b64_image)
        image_bytes = BytesIO(decoded_data)
        image = PIL_Image.open(image_bytes).convert('RGB')
        
        filename = Imagen2.new_filename()
        image.save(filename) 
        
        return filename

    
    @staticmethod
    def new_filename():
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            filename = temp_file.name
        return filename
    
    
    @staticmethod
    def image_to_b64(image):
        
        filename = Imagen2.new_filename()
        image.save(filename)
            
        with open(filename, "rb") as image_file: 
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            
        return encoded_string
        
    
    def generate_background_b64(self, prompt, b64_image, number_of_images=5):
        image_path =  self.b64_to_file(b64_image)
        return self.generate_background(prompt, image_path, number_of_images)
    
                 
    def generate_background(self, prompt, image_path, number_of_images=5):
            
            mask_object = self.segmentation_util.segment_image_to_file(image_path)
            mask_filename = mask_object.get('filename')

            base_img = Image.load_from_file(location=image_path)
            mask_img = Image.load_from_file(location=mask_filename)
            
            images = self.run_imagen_edit(prompt, base_img, mask_img, number_of_images=5)
            
            return images


    @APSX_Logger.log_and_time
    def run_imagen_edit(self, prompt, base_img, mask_img, number_of_images=5):

        images = self.imagen_model.edit_image(
            prompt=prompt,
            base_image=base_img,
            mask=mask_img,
            # mask_mode='background',
            mask_dilation = 0,
            
            # Optional parameters
            # seed=1,
            # Controls the strength of the prompt.
            # -- 0-9 (low strength), 10-20 (medium strength), 21+ (high strength)
            guidance_scale=15,
            number_of_images=number_of_images,
            safety_filter_level="block_few")
                 
        return images
