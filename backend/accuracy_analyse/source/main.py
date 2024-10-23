import os
import functions_framework

import firebase_admin
from firebase_admin import firestore

import APSX_Firestore_Utils

import APSX_Accuracy_Analyser



app, db = APSX_Firestore_Utils.init()
    
# load analyser helper     

# load analyser helper     
predict_every_nth_frame = int(os.environ.get('predict_every_nth_frame', 4))
confidence_threshold = float(os.environ.get('confidence_threshold', 0.5))

print(f'⚙️ predict_every_nth_frame : {predict_every_nth_frame}')
print(f'⚙️ confidence_threshold : {confidence_threshold}')

analyser = APSX_Accuracy_Analyser.Accuracy_Analyser(ball_model_path='berlin_ball_back.tflite', 
                                                    goal_model_path='berlin_goal.tflite',
                                                    bins_model_path='bins.tflite',
                                                    every_nth_frame=predict_every_nth_frame, 
                                                    confidence_threshold=confidence_threshold)

# top left and right corners
DEFAULT_TARGETS = [[0,0], [1,0]]


@functions_framework.http
def analyse(request):
    """
    get_json 
        {'file_name': name,
           'event_id': event_id,
           'session_id': session_id,
           'action_id': action_id,
           'camera_id': camera_id,
           'time': data["timeCreated"]}
    """
    request_json = request.get_json()
    
    # get data from request  
    file_name = request_json.get('file_name')
    action_id = request_json.get('action_id')
    event_id = request_json.get('event_id')
    print(request_json)
    
    # run analysis     
    run_analysis_and_update_firestore(file_name, action_id, event_id, debug=False)
    
    return '✅ Analysed accuracy for {}'.format(file_name)


def run_analysis_and_update_firestore(file_name, action_id, event_id, debug):
    
    # get action doc     
    action_ref = db.collection("actions").document(action_id)
    
    # get targets
    targets = get_targets(action_ref)
    
    # get accuracy score     
    accuracy_score, accuracy_metadata = analyser.analyse(file_name, targets, event_id, debug=debug) 
    print(f'accuracy score {accuracy_score}, metadata: {accuracy_metadata}')
    
    # update firestore
    action_ref.update({"accuracy":{"score":accuracy_score, "metadata":accuracy_metadata}})
    print(f'updated action id {action_id}')
#     
def get_targets(action_ref):
    
    '''
    Get the normalised positions of the targets used to calculate the accuracy score.
    [[x,y],[x,y]...] x and y are floats from 0 to 1
    '''
    
    action_data = action_ref.get().to_dict()
    
    metadata = action_data.get('metadata', {})

    targets = DEFAULT_TARGETS
    
    if 'target' in metadata:
        targets = [[metadata['target']['x'], metadata['target']['y']]]
        print('using defined target', targets)
    else:
        print('using DEFAULT target', targets)
        
    return targets
    
