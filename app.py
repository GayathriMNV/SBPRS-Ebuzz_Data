from flask import Flask, jsonify,  request, render_template
import numpy as np
import pandas as pd
import pickle as pickle
from model import top_20_recommended_products,top_5_products_using_sentiment_analysis

import os

#to get the current working directory
dir = os.getcwd() + "/" + 'nltk_data'
print(dir)

import nltk
nltk.data.path.append(dir)

app = Flask(__name__)

clean_data = pickle.load(open('reviews_clean_data.pkl', 'rb'))
tfidf_vect_model = pickle.load(open('tfidf.pkl', 'rb'))
smote_model = pickle.load(open('smote.pkl', 'rb'))
user_recommendation_model = pickle.load(open('user_based_sentiment_model.pkl', 'rb'))
logistic_model = pickle.load(open('LogisticRegression_Model.pkl', 'rb'))

@app.route('/')

def home():
	return render_template('index.html')

@app.route("/predict", methods = ['POST'])

def predict():
	if( request.method == 'POST'):
		user_name = request.form['reviews_username']
		user_name = user_name.lower()

		# If user_name doesnot exist in the dataset
		if user_name not in user_recommendation_model.index :
			return render_template('index.html', prediction_text = 'Enter a valid User Name')

		top5_products_recommended = top_5_products_using_sentiment_analysis(user_name)

		return render_template('index.html',  user_name = user_name, prediction_text = '1.{} 2.{} 3.{} 4.{}5.{}'.format(top5_products_recommended[0],
															       top5_products_recommended[1],
														               top5_products_recommended[2],
															       top5_products_recommended[3],
															       top5_products_recommended[4],
															       )
					)
			
	else:
		return render_template('index.html')	


if __name__ == "__main__":
	app.run(debug = True)
