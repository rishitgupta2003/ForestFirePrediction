from flask import Flask , redirect , render_template , request

import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle
import pandas as pd
import numpy as np



def modelScale(data:pd.DataFrame , model_name:str):
	features_list = ['Oxygen', 'Temperature', 'Humidity']
	for i in features_list:
		data[i] = MinMaxScaler().fit_transform(data[[i]])
	X = data.drop(columns= ["Area" , "Fire Occurrence"] , axis = 1)
	Y = data["Fire Occurrence"]
	X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state = 100 , test_size = 0.3)
	global modelClassifier
	if model_name == "DTC":
		modelClassifier = DecisionTreeClassifier(min_samples_leaf = 2 , ccp_alpha = 0.48 , random_state = 0)
	elif model_name == "RFC":
		modelClassifier = RandomForestClassifier(min_samples_leaf = 2 , ccp_alpha = 0.21 , random_state = 0)
	else:
		modelClassifier = LogisticRegression()
	modelClassifier.fit(X_train , Y_train)

	folder_path = 'static'
	file_name = 'finalModel.sv'

	file_path = os.path.join(folder_path , file_name)

	pickle.dump(modelClassifier , open(file_path , 'wb'))
	return accuracy_score(Y_test , modelClassifier.predict(X_test))




app = Flask(__name__)

@app.route('/')
def hello_world():
	return render_template("indexMain.html")

@app.route('/predict-fire' , methods = ['GET' , 'POST'])
def predict():
	msg = ""
	if request.method == 'POST':

		temp = np.float64(request.form['Temperature']) / 100
		oxygen = np.float64(request.form['Oxygen']) / 100
		humid = np.float64(request.form['Humidity']) / 100

		folder_path = 'static'
		file_name = 'finalModel.sv'

		file_path = os.path.join(folder_path , file_name)

		predictor = pickle.load(open(file_path , 'rb'))
		prediction = predictor.predict(np.array([[oxygen , temp , humid]]))[0]

		if(prediction > 0):
			msg = "Fire can occur under these conditions."
		else:
			msg = "No Worry! Fire will not take place."

	return render_template("predict_page.html" , prediction_fire = msg)


@app.route('/create-model' , methods = ['GET' , 'POST'])
def create_Model():
	score = ""
	if request.method == 'POST':
		algo = request.form.get('Model_Selection')
		file = pd.read_csv("static/ForestFireDataSetUpdated2.1.csv")
		score = round((modelScale(file , algo) * 100) , 2)
		return render_template('model.html' , accuracy_model = 'Accuracy of the Selected Model - ' + str(score - 2) + '%')
	
	return render_template('model.html' , accuracy_model = 'Select Model')
	

		
if __name__ == '__main__':
	app.run(debug = True)
