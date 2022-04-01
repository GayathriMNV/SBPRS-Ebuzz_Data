
'''
# Sentiment-Based Product Recommendation system, which includes the following tasks,

1.Data sourcing and sentiment analysis
2.Building a recommendation system
3.Improving the recommendations using the sentiment analysis model
4.Deploying the end-to-end project with a user interface

'''
#  Importing packages,Reading & Understanding Data

import pandas as pd
import nltk
nltk.data.path.append('./nltk_data/')
import re
import pickle as pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# Importing libraries to balance Class
from collections import Counter
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Load reviews clean data,tfidf, LogisticRegression and User Based Recommendation pickle files

clean_data = pickle.load(open('reviews_clean_data.pkl', 'rb'))
tfidf_vect_model = pickle.load(open('tfidf.pkl', 'rb'))
smote_model = pickle.load(open('smote.pkl', 'rb'))
user_recommendation_model = pickle.load(open('user_based_sentiment_model.pkl', 'rb'))
logistic_model = pickle.load(open('LogisticRegression_Model.pkl', 'rb'))


def normalize_text(text):
    
    #Make the text lowercase
    text_lower = text.lower()
     
    # To remove 1), 2) like paterns 
    text_no_curve = re.sub(r'^\d[1]\)', '',text_lower) 
    
    # HTML Tags removal
    text_no_html = re.sub(r'<.*?>' , '', text_no_curve)
        
    #Remove punctuation & special character
    text_nospl = re.sub(r'[?|!|\'|"|#|.|,|)|(|\|/|~|%|*-]', '', text_no_html)
    
    # Convert the numbers to strings
    text_str = str(text_nospl)
    
    return text_str

# Lets a write a function to lemmatize the reviews text
def  lemmatize_text(text):
    stopwords_list = set(stopwords.words('english'))
    lm = WordNetLemmatizer()
    tokenised_text = nltk.word_tokenize(text)
    lemmatised_text = [lm.lemmatize(word, pos = 'v') for word in tokenised_text if word not in stopwords_list]
    clean_text = ' '.join(lemmatised_text)
    return clean_text


def normalize_lemmatize(text):
    input_text = normalize_text(text)
    output_text = lemmatize_text(input_text)
    return output_text

# Write inside fun 
def train_tf_model(text):
    tf_text = tfidf_model.transform(text)
    return tf_text


def build_lr_model(text):
    y_pred = logistic_model.predict(text)
    return y_pred

def model_prediction(text):
    clean_text = normalize_lemmatize(text)
    tf_text = train_tf_model(clean_text)
    ml_predict = build_lr_model(tf_text)
    return ml_predict

# Lets create a fucntion to print the top 20 recommendations for an user
def top_20_recommended_products(user_name):

    user_top20_products = user_recommendation_model.loc[user_name].sort_values(ascending=False)[:20]
    user_top20_products = pd.DataFrame(user_top20_products)  #.to_records())
    user_top20_products.reset_index(inplace = True)
    # merge top 20 products and its reviews
    top20_products_sentiment = pd.merge(user_top20_products,clean_data,on = ['name'])
    # convert text to feature
    top20_products_tfidf = tfidf_vect_model.transform(top20_products_sentiment['processed_reviews'])
    # model prediction
    top20_products_recommended= logistic_model.predict(top20_products_tfidf)
    #top20_products_recommended= model_prediction(top20_products_sentiment['processed_reviews'])
    top20_products_sentiment['top20_products_pred'] = top20_products_recommended
    sentiment_score = top20_products_sentiment.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
    sentiment_score['percent'] = round((100*sentiment_score['sum'] / sentiment_score['count']),2)
    
    return sentiment_score.head(20)

# Lets define a function to print the top 5 recommendations to an User
def top_5_products_using_sentiment_analysis(user):
    top_20_prods= top_20_recommended_products(user)
    sentiment_score = top_20_prods.sort_values(by='percent',ascending=False)
    top5_products_recommended = sentiment_score['name'].head().to_list()
    return top5_products_recommended

#top_5_products_using_sentiment_analysis('aaron')