
import json

import APSX_Logger
import APSX_Video_Processor
import APSX_GCS_utils
import APSX_Player_Segmentation
import APSX_Gemini_Multimodal

from vertexai.generative_models import Image


class Gemini_Style_Analyser_Images():
    
    
    def __init__(self):
            self.model = APSX_Gemini_Multimodal.Gemini_Multimodal("gemini-1.5-pro-001")
            self.segmentor = APSX_Player_Segmentation.PlayerSegmentation()
            print('✅ Loaded ', self)

            
    @APSX_Logger.log_and_time 
    def analyse(self, gs_file_name, debug=False):
        
        local_video_filename = APSX_GCS_utils.download_from_gcs(gs_file_name)
        
        frames = APSX_Video_Processor.extract_frames_at_regular_intervals(local_video_filename, 20)[5:-5]
        cropped_frames =  self.crop_numoy_frames_to_person(frames)
        images_for_gemini = self.numpy_images_to_gemini_images(cropped_frames)
        
        prompt = self.build_prompt_list(images_for_gemini)
        model_response = self.model.call_gemini(prompt, parse_json=True)
        
        return {'model_response': model_response, 'numpy_frames': cropped_frames, 'p':prompt}

    
    def crop_numoy_frames_to_person(self, numpy_frames):
        return [self.segmentor.crop_numpy_to_person(f) for f in numpy_frames]
    
    def numpy_images_to_gemini_images(self, numpy_frames):
        return [Image.from_bytes(APSX_Video_Processor.numpy_image_to_bytes(frame)) for frame in numpy_frames ]


    def build_prompt_list(self, gemini_images: list[Image]) -> list:
        
        
        prompt_start = '''
        Carefully analyze the provided soccer penalty kick images from the player angle and generate a "style score" based on their technique with the following considerations:

Technique: Did the player exhibit interesting or unorthodox kicking technique, if so give them alot of points.  If they are wearing brightly coloured clothing (white included) put them in the 80s.

Scoring guide:

Base score
- standing still: 50-60
- slow run up : 60-70
- energetic run up : 80-90
- celebration dance, backwards kick, kicking ball from hand, spinning, throwing ball   : 90-99

Bonus score
- plain dark clothing : +0-10 bonus points
- brightly colours : +10-20 bonus points
- bold patterns (e.g animal print, strips) : instant 99 max points

Total score = base + bonus

And the total score maxing out at 99 (even if it calculates to be over 99)

Also Your job is to pick the best "hero" moment of someone kicking a soccer ball from the following photos.
        Top tip for conventional kicks: Select the moment where the kicking leg is fully cocked back (the leg wind up moment that shows "the power of the kick") just before the player begins to swing their kicking leg through. 
        For unconventional kicks just pict the photo that looks the coolest.
        Here is teh sequence of photos to choose from (use a zero base index so the first image is "0", and the second is "1", etc)

        Images:
        
        '''
        
        
#         prompt_start = '''
#         You job is to pick the best "pre kick" moment of someone kicking a soccer ball from the following photos.
#         Select the moment where the kicking leg is fully cocked back (the leg wind up moment that shows "the power of the kick") just before the player begins to swing their kicking leg through. 

#         Also give a style score from 50 -> 99 for how cool their cloths are i,e high score for bright fashonable colours 

#         '''
        parts = []
        for (i, photo) in enumerate(gemini_images):
            # parts.append(f'Photo {i}:')
            parts.append(photo)
#         prompt_end = '''from the above list pick the best photo (use a zero base index so the first image is "0", and the second is "1", etc) and judge their style and give your detailed reasoning for both and output just a json with the properties:
#         "picked_photo_index" -> int 
#         "picked_photo_reasoning" -> string 
#         "style_score" -> int 
#         "style_score_reasoning" -> string 

#         json output: 
#         '''
        
        
        prompt_end = '''
        Provide your analysis in the following JSON format:

JSON Output example:

{
  "clothing_description": string // description of clothing
  "kicking_technique_descirption": string // description of the kicking action seen in the video
  "style_score_reasoning": string  // explanation of score
  "style_score": int,  // the total style score from 50 to 99

   # picked photo / hero photo 
  "picked_photo_reasoning": string 
  "picked_photo_index": int
}
        json output: 

        '''
        
        return [prompt_start, *parts, prompt_end]
        
