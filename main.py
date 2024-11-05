import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("LinearModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    bhk_options = sorted(data['bhk'].unique())  # Unique BHK options from the dataset
    bath_options = sorted(data['bath'].unique())  # Unique bathroom options from the dataset
    return render_template('index.html', locations=locations, bhk_options=bhk_options, bath_options=bath_options)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get("location")
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Prepare input DataFrame
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    # Predict price
    prediction = pipe.predict(input_data)[0] * 1e5  

    return f"{np.round(prediction, 2)}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
