
##require google-cloud-aiplatform

import json

import APSX_Logger

from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models



class Gemini_Multimodal():
    
    def __init__(self, model_name = "gemini-1.5-flash-001"):
            self.model = GenerativeModel(model_name)
            print('✅ Loaded ', model_name , self)
    
    
    @APSX_Logger.log_and_time 
    def call_gemini(self, prompt_list, parse_json=True, temperature=0.5):
        
        
        generation_config = {
            "temperature": temperature,
        }
        
        
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        model_response = self.model.generate_content(prompt_list, 
                                                     generation_config=generation_config,
                                                     safety_settings=safety_settings)
        
        print("raw model response: ", model_response.text)
        
        if parse_json:
            return self.extract_json(model_response.text)
        
        return model_response.text
    
    
    @staticmethod
    def extract_json(string):
        """
        Extracts pure, clean JSON from within a string.

        Args:
        string: The string containing the JSON.

        Returns:
        The extracted JSON object (as a Python dictionary), or None if no valid JSON is found.
        """

        # Attempt to find the start and end of the JSON object.
        start_index = string.find('{')
        end_index = string.rfind('}')

        # If valid indices are found, extract the JSON.
        if start_index != -1 and end_index != -1:
            json_string = string[start_index:end_index + 1]
        else:
            return {'output': string}


        # Attempt to parse the JSON string.
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print("Invalid JSON found within the string.")
            return None

        else:
            print("No valid JSON found within the string.")
            return None
        
