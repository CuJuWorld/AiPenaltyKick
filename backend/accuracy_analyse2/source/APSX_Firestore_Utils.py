
##require firebase-admin

import firebase_admin
from firebase_admin import firestore

def init():

    if not firebase_admin._apps:
        app = firebase_admin.initialize_app()
        db = firestore.client()
    else:
        app  = firebase_admin.get_app()
        db  = firestore.client()
        
    return app, db
            
