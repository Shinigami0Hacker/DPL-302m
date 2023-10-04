from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import os
from Model import AIModel
app = Flask(__name__)

input_direction = "./input"

cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/upload_text", methods = ['POST', 'GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def upload_text():
    """
    @return
    Content: The respond content.
    Status code: The respond status code of HTTP.
    """
    data = json.loads(request.data.decode('utf-8'))
    with open(os.path.join(input_direction, "input.json"), "w") as jfile:
        json.dump(data, jfile)

    model = AIModel.Model()
    
    prediction = model.predict()

    result = []
    for i in range(len(prediction)):
        result.append((data[i], prediction[i]))
    
    result = json.dumps(result)
    return result, 200

if __name__ == '__main__':
    app.run(debug=True)