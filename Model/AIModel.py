import json
import numpy as np

labeled = [
    "Negative",
    "Positive",
    "Neural"
]

class Model:
    def __init__(self) -> None:
        self.input_direction = "./input/input.json"
        with open(self.input_direction, "r") as file:
            self.data = json.load(file)
    
    def predict(self):
        mapping_matrix = [0] * len(self.data)
        for i in range(len(self.data)):
            result = np.random.choice(labeled)
            mapping_matrix[i] = result
        return mapping_matrix