from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predic():
    feature = request.form['features']
    feature_list = feature.split(',')
    np_features = np.asarray(feature_list, dtype=np.float64)
    predict = model.predict(np_features.reshape(1, -1))

    output = ["Cancerous" if predict[0] == 1 else "Not Cancerous"]

    return render_template("index.html", message=output)



if __name__ == "__main__":
    app.run(debug=True)
