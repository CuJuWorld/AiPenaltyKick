import datetime
import threading
import requests
import functions_framework
        
import google.auth.transport.requests
import google.oauth2.id_token

import firebase_admin
from firebase_admin import firestore

if 'app' not in globals():
    app = firebase_admin.initialize_app()
    db = firestore.client()

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def new_file(cloud_event):
    
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]
    
    print(f'new file {bucket} {name} ')
    
    # get the meta data from the file name     
    (event_id, session_id, action_id, camera_id) = extract_file_data(name)
    
    file_name = f'{bucket}/{name}'
    
    # assemble data package to be sent to routed cloud function     
    data_to_send = {'file_name': file_name,
           'event_id': event_id,
           'session_id': session_id,
           'action_id': action_id,
            'camera_id': camera_id,
            'time': data["timeCreated"]}
    print(f'event_id: {event_id}, session_id :{session_id}, action_id: {action_id}, camera_id: {camera_id}')
    
    # cerate recording doc     
    create_recording_doc(action_id, camera_id, file_name)
    
    # get the processing configs for the camera which will contain the 
    # function names to send the videos to to be processed      
    camera_data = db.collection("cameras").document(camera_id).get().to_dict()
    video_processing_configs = camera_data.get('processing_configs', [])
    
    print(video_processing_configs)
    
    threads =[]
    
    # for each processing config, send the video data to that cloud function     
    for config in video_processing_configs:
        
        config_data = config.get()
        function_name = config_data.get('function_name')
        
        print('calling: ', function_name)
        
        thread = threading.Thread(target=call_cloud_function, args=(function_name, data_to_send))
        thread.start()
        threads.append(thread)
        print(f'File {name} directed to {function_name} with data {data_to_send}')
        
        # call_cloud_function(function_name, data_to_send)
    
    # wait for all to return     
    for t in threads:
        t.join()
        
def create_recording_doc(action_id, camera_id, file_name):
    '''
    create a recording record for the data
    '''
    data = {
     "action_id": action_id,
     "camera": db.collection("cameras").document(camera_id),
     "camera_id": camera_id,
     "file_location": file_name,
     "time_created" : datetime.datetime.now() 
    }
    
    recording_id = action_id+camera_id
    db.collection("recordings").document(recording_id).set(data)
    
    
    
def extract_file_data(file_name):
    '''
    Extract out the video data from the file name.
    '''
    just_name = file_name.split('.')[0]
    name_parts = just_name.split('_')
    event_id, session_id, action_id, camera_id = name_parts
    
    return (event_id, session_id, action_id, camera_id)


def call_cloud_function(url, data):
    
    # get auth token to call cloud function     
    token = get_id_token(url)

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for non-2xx responses
        # Process the response from the Cloud Function
        print(response.text)

    except requests.exceptions.RequestException as e:
        # Handle any request-related errors
        print(f"Error sending request: {e}")


def get_id_token(endpoint):
    audience = endpoint
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
    return id_token
