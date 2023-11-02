import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model, sc = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)

    final_features = sc.transform(final_features)
    prediction = model.predict(final_features)

    if prediction == 1:
        return render_template('index.html',
                               prediction_text = "Yes! this customer would have bought the product")

    if prediction == 0:
        return render_template('index.html',
                               prediction_text = "Unfortunately this customer would not have bought the product")

if __name__ == "__main__":
    app.run(debug=True)