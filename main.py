from joblib import load
from klein import run, route
import json
from preprocess import Preprocess
import numpy as np

model = load('model.pkl')
prep = Preprocess()
#labels = {0: 'Negative', 1: 'Positive', 2: 'Hostil'}

@route('/', methods=['POST'])
def do_post(request):
  content = json.loads(request.content.read())
  X = np.array([content['text']])
  X = prep.run(X)
  y = model.predict(X)
  return json.dumps({"sentiment":y[0]}, indent=4)

run("localhost", 8080)
