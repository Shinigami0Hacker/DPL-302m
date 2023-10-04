input_direction = "./input"
import json
from Model import AIModel
def upload_text():
    """
    @return
    Content: The respond content.
    Status code: The respond status code of HTTP.
    """
    # data = json.loads(request.data.decode('utf-8'))
    # with open(os.path.join(input_direction, "chat_content.json"), "w") as jfile:
    #     json.dump(data, jfile)
    with open("./input/input.json", "r") as file:
        data = json.load(file)
    model = AIModel.Model()
    
    prediction = model.predict()

    result = []
    for i in range(len(prediction)):
        result.append((data[i]['chat_content'],prediction[i]))
    return result

s = upload_text()
print(s)