
"""Utility functions for handling CORS and authentication in Firebase Cloud Functions for web apps.

This module provides helper functions to streamline CORS configuration and user
authentication in Google Cloud Functions triggered by requests from Firebase web apps.

Functions:
  check_CORS_and_AUTH(request):
    Checks for CORS preflight requests and handles user authentication using a Firebase
    Authentication token. Returns a dictionary containing CORS headers, user data
    (if authenticated), error response (if authentication fails), and response code.

  get_cors_headers():
    Returns a dictionary of CORS headers allowing requests from all origins
    (adjustable), defining allowed methods and headers, and setting a cache duration.

  authenticate_user(request):
    Extracts the Firebase Authentication token from the request data and verifies it.
    Returns the decoded token (containing user information) on success or an error
    response dictionary on failure.
"""

##require firebase-admin

import firebase_admin
from firebase_admin import auth


if 'app' not in globals():
    app = firebase_admin.initialize_app()
    
    
class CorsAuthResponse:
    """Helper class to represent a response object with common properties."""

    def __init__(self):
        self.return_now = False
        self.user = None
        self.body = ''
        self.headers = get_cors_headers()
        self.response_code = None
        
    def to_dict(self):
        """Returns a dictionary representation of the Response object."""
        return {
            "return_now": self.return_now,
            "user": self.user,
            "body": self.body,
            "headers": self.headers,
            "response_code": self.response_code,
        }

    def __str__(self):
        """Overrides the default string representation to print as a dictionary."""
        return str(self.to_dict())
    
    def http_response(self):
        
        return (self.body, 
                self.response_code, 
                self.headers)

    
def check_required_fields(request, required_fields_list=None):
    
    if required_fields_list is None:
        required_fields_list = []
    
    data = request.get_json().get('data', {})
    missing_fields = [field for field in required_fields_list if (not data.get(field, False))]
    return missing_fields
        
    
def check_CORS_Auth_Fields(request, required_fields_list=None):
    """
    user object keys = ['name', 'picture', 'iss', 'aud', 'auth_time', 'user_id',  
    'sub', 'iat', 'exp', 'email', 'email_verified', 'firebase', 'uid']
    """

    output = CorsAuthResponse()
    if request.method == 'OPTIONS':
        output.return_now = True
        output.response_code = 204
        return output

    user, error_response = authenticate_user(request)
    if error_response:
        output.return_now = True
        output.response_code = 401
        output.body = error_response
        return output
    
    missing_fields = check_required_fields(request, required_fields_list)
    if missing_fields:
        output.return_now = True
        output.response_code = 400
        message = f"Missing required field(s): {', '.join(missing_fields)}"
        output.body = {'data': {'message': message}}
        return output

    output.response_code = 200
    output.user = user

    return output


def get_cors_headers():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type, authorization",
        "Access-Control-Max-Age": "3600"
    }
    return headers


def authenticate_user(request):
    token = request.get_json(silent=True).get('data', {}).get('id', None)

    if token is None:
        return None, {'message': 'Missing Firebase Authentication token', 'code': 401}

    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token, None
    except Exception as error:
        print(f'Firebase Authentication failed: {error}')
        return None, {'message': 'Unauthorized', 'code': 401}
