

import APSX_Firestore_Utils

app, db = APSX_Firestore_Utils.init()
    
DEFAULT_VIDEO_BUCKET = 'apsx-videos'
DEFAULT_MISC_DATA_BUCKET = 'apsx-misc-data'
    
cache = {}
    
class APSX_Event_Info_Helper():
    
    def __init__(self):
        pass
    
    def get_info(self, event_id):

        if event_id is None:
            return {}
        
        if event_id not in cache:
            cache[event_id] = db.collection("events").document(event_id).get().to_dict()
        else:
            print('using cache for ', event_id)
        data = cache[event_id]
        return data
    
    def get_misc_data_bucket(self, event_id):
        data = self.get_info(event_id)
        misc_data_bucket = data.get('misc_data_bucket', DEFAULT_MISC_DATA_BUCKET)
        return misc_data_bucket
    
    def get_video_bucket(self, event_id):
        data = self.get_info(event_id)
        video_upload_bucket = data.get('video_upload_bucket', DEFAULT_VIDEO_BUCKET)
        return video_upload_bucket
