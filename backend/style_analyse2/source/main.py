import functions_framework

import firebase_admin
from firebase_admin import firestore

import APSX_GCS_utils
import APSX_Video_Processor
import APSX_Gemini_Style_Analyser_Images

import APSX_Event_Info_Helper
import APSX_Firestore_Utils


app, db = APSX_Firestore_Utils.init()
event_info_helper = APSX_Event_Info_Helper.APSX_Event_Info_Helper()


analyser = APSX_Gemini_Style_Analyser_Images.Gemini_Style_Analyser_Images()

@functions_framework.http
def analyse(request):
    """
    get_json 
        {'file_name': name, "apsx-videos/1dzI9_mLrd_6YyYS_5327ce.mp4"
           'event_id': event_id, "1dzI9"
           'session_id': session_id, "mLrd"
           'action_id': action_id, "6YyYS"
           'camera_id': camera_id, "5327ce"
           'time': data["timeCreated"]} 
    """
    
    request_json = request.get_json()
    
    # look at sent data     
    file_name = request_json.get('file_name')
    action_id = request_json.get('action_id')
    event_id = request_json.get('event_id')

    print(request_json)
    
    # perform analysis sycronously
    run_analysis_and_update_firestore(file_name, action_id, event_id)

    return 'Analysed style: {}'.format(file_name)


def run_analysis_and_update_firestore(file_name, action_id, event_id):
    
    # get style score     
    analysis_output = analyser.analyse(file_name) 
    
    analysis_json = analysis_output.get('model_response')    
    style_score = analysis_json.get('style_score', 66)
    
    numpy_frames = analysis_output.get('numpy_frames')
    gcs_uris = save_all_frames_to_gcs(numpy_frames, action_id, event_id)
    
    top_pick_frame_uri = gcs_uris[analysis_json.get('picked_photo_index', 8)]
    
    new_action_data = {**analysis_json, 
                       'top_pick_image' : top_pick_frame_uri ,
                        'all_images': gcs_uris}
    
    update_firestore_action(action_id, new_action_data)
    

    

def update_firestore_action(action_id, data):
    
    action_ref = db.collection("actions").document(action_id)
    
    image_frame_url = data.get('top_pick_image')
    style_score = data.get('style_score')

    metadata = {
    'pose_plot': image_frame_url,
    'kick_frame': image_frame_url,
    'prediction_count': 10,
    **data
    }
    
    updated_data = {"style": {'score':style_score, 'metadata': metadata}}
    
    action_data = action_ref.get().to_dict()
    power_score = action_data.get('power',{}).get('score', None)
    accuracy_score = action_data.get('accuracy',{}).get('score', None)

    if (power_score and accuracy_score):
        print('overwriting total_score')
        total_score = sum([power_score, accuracy_score, style_score])//3
        updated_data['total_score'] = total_score
    
    action_ref.update(updated_data)
    
    print(f'updated action id {action_id}')

    
    
def save_all_frames_to_gcs(numpy_frames, action_id, event_id):
    
    asset_bucket  = event_info_helper.get_misc_data_bucket(event_id)

    gcs_uris = []
    
    for index, frame in enumerate(numpy_frames):
        frame_bytes = APSX_Video_Processor.numpy_image_to_bytes(frame)
        frame_gcs_uri = f'{asset_bucket}/{action_id}/all_style_frames/{index}.jpg'
        APSX_GCS_utils.upload_bytes_to_gcs(frame_bytes, frame_gcs_uri)
        gcs_uris.append(frame_gcs_uri)

    return gcs_uris
    
