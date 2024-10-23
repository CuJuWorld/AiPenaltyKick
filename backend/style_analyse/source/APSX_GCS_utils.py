
import tempfile
from google.cloud import storage

import APSX_Logger


@APSX_Logger.log_and_time
def copy_blob(bucket_name, blob_name, destination_bucket_name, destination_blob_name):
    """Copies a blob from one bucket to another with a new name."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # destination_bucket_name = "destination-bucket-name"
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(source_blob, 
                                        destination_bucket, 
                                        destination_blob_name)
    
    print(f'✅ Copied {bucket_name}/{blob_name} to {destination_bucket_name}/{destination_blob_name}')



@APSX_Logger.log_and_time
def upload_to_gcs(local_filename, gs_location):
    """Uploads an image to a Google Storage bucket.

    Args:
    bucket_name: The name of the bucket.
    filename: The path to the file.
    blob_name: The name of the blob in the bucket.
    """
    
    bucket_name, blob_name = gs_location.replace('gs://', '').split('/', 1)
    
    print(f'Uploadting {local_filename} to bucket:{bucket_name}, blob {blob_name} ...')

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_filename)
    print(f'✅ Uploaded {local_filename} to bucket:{bucket_name}, blob {blob_name}')
    

@APSX_Logger.log_and_time
def upload_bytes_to_gcs(object_bytes, gs_location, content_type='image/jpeg'):
    """Uploads bytes as an image file to Google Cloud Storage.

    Args:
    """
    
    bucket_name, blob_name = gs_location.replace('gs://', '').split('/', 1)
    
    print(f'Uploadting bytes to bucket:{bucket_name}, blob {blob_name} ...')

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(object_bytes, content_type=content_type)
    print(f'✅ Uploaded bytes to bucket:{bucket_name}, blob {blob_name}')
    

@APSX_Logger.log_and_time
def download_from_gcs(gs_location):
    """Loads a file from Google Storage.
    """
    print(f'Downloading file :{gs_location}...')
    
    bucket_name, blob_name = gs_location.replace('gs://', '').split('/', 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        blob.download_to_file(temp)
    print(f'✅ Downloaded file :{gs_location}')
    return temp.name


def set_access_uid_metadata_gcs(gs_location, access_uid):

    print(f'Setting access UID metadata :{gs_location} -> {access_uid}')
    
    bucket_name, blob_name = gs_location.replace('gs://', '').split('/', 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.metadata = {'access_uid': access_uid}
    blob.patch()

    print(f'✅ Metadata access UID updated!')
    return True


class APSX_File_Name():

    def __init__(self, gs_path):
        self.raw_path = gs_path
        self.path = gs_path.replace('gs://','').split('/')[-1]
        just_name = self.path.split('.')[0]
        name_parts = just_name.split('_')
        self.event_id, self.session_id, self.action_id, self.camera_id = name_parts
