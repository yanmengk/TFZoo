import json
import numpy as np
import requests
from tf_wiki.basic.ex04_MLP_mnist import MNISTLoader


data_loader = MNISTLoader()
data = json.dumps({
    "instances": data_loader.test_data[0:10].tolist()
    })

headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/mnist_test2:predict',
    data=data, headers=headers)

predictions = np.array(json.loads(json_response.text)['predictions'])
print(np.argmax(predictions, axis=-1))
print(data_loader.test_label[0:10])