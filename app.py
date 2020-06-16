# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'insulin-resistance-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = float(request.form['age'])
        waist = float(request.form['waist circumference'])
        bloodpressure = float(request.form['bloodpressure'])
        h1 = height/100
        waistsize = waist*2.54
        bmi = weight/h1*h1
        AGE_BMI = age/bmi
        
        data = np.array([[waistsize,bloodpressure,AGE_BMI]])
        print(data)
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)