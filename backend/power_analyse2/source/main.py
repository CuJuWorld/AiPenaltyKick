
import os

import functions_framework

import firebase_admin
from firebase_admin import firestore

import APSX_Firestore_Utils

import APSX_Power_Analyser


app, db = APSX_Firestore_Utils.init()


# load analyser helper     
predict_every_nth_frame = int(os.environ.get('predict_every_nth_frame', 4))
confidence_threshold = float(os.environ.get('confidence_threshold', 0.5))

print(f'⚙️ predict_every_nth_frame : {predict_every_nth_frame}')
print(f'⚙️ confidence_threshold : {confidence_threshold}')

analyser = APSX_Power_Analyser.Power_Analyser(model_path='berlin_side.tflite',
                                              every_nth_frame=predict_every_nth_frame,
                                              confidence_threshold=confidence_threshold)

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
    
    run_analysis_and_update_firestore(file_name, action_id, event_id)

    return 'Analysing {}'.format(file_name)


def run_analysis_and_update_firestore(file_name, action_id, event_id):
    
    # get power score     
    power_score, power_metadata = analyser.analyse(file_name, event_id, debug=False) 
    print(f'power score {power_score}, metadata: {power_metadata}')
    
    # update firestore
    action_ref = db.collection("actions").document(action_id)
    action_ref.update({"power": {"score":power_score,
                                  "metadata":power_metadata}})
    print(f'updated action id {action_id}')
    
    
