from flask import Flask, request, Response, jsonify
import numpy as np
from cv2 import cv2
import main
import base64
# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        req = request.json
        # decode base64 string into np array
        nparr = np.frombuffer(base64.b64decode(req['image'].encode('utf-8')), np.uint8)
        # decoded image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        user_id, user_name, user_birthdate, user_data = main.ocr(img)
        response = {
            'id': user_id,
            'name': user_name,
            'birthdate': user_birthdate,
            'data': user_data
        }
        resp = jsonify(response)
        return resp
    except Exception as e:
        response = {
            'success': False,
            'status code': 500,
            'message': str(e),
        }
        resp = jsonify(response)
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    # start flask app
    app.run(threaded=True)
