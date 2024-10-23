
import functions_framework

from PIL import UnidentifiedImageError

import APSX_HTTP_Utils
import APSX_Imagen2

imagen = APSX_Imagen2.Imagen2()

REQUIRED_FIELDS = ['prompt', 'b64_image']

@functions_framework.http
def generate(request):
    """A Cloud Function that processes request JSON data."""
    
    response = APSX_HTTP_Utils.check_CORS_Auth_Fields(request, REQUIRED_FIELDS)

    if response.return_now:
        return response.http_response()
    
    print('Passed CORS, Auth, and required fields check  ✅')
    
    # user = reponse.user
    request_data = request.get_json().get('data', {})
    
    prompt = request_data.get('prompt', '')
    b64_image = request_data.get('b64_image', '')

    try:
        images = imagen.generate_background_b64(prompt, b64_image)
    except UnidentifiedImageError:
        print('❌ invalid b64 image data')
        response.response_code = 400
        response.body = {'data': {'message': "invalid b64 image data"}}
        return response.http_response()

    
    b64_image_list = [imagen.image_to_b64(i) for i in images]
    
    response.body = {'data': {'images': b64_image_list}}
    return response.http_response()


