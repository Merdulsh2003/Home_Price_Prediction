from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
data= pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl","rb"))
@app.route('/images/<filename>')
def serve_image(filename):
    return send_file('images/homes.jpg')
@app.route('/')
def index():

    locations =sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    total_sqft = float(request.form.get('total_sqft'))

    print(location, bhk, bath, total_sqft)
    input_data = pd.DataFrame([[location,total_sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input_data)[0]*1e5
    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)