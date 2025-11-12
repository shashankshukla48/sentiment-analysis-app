#import essential libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk 
from nltk.corpus import stopwords #to remove unnecessary words
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


#load the trained regression model
try:
    model = joblib.load('lr_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model not found")
    model = None
    vectorizer = None
except Exception as e:
    print('An error occured in loading the file : {e}')
    model = None
    vectorizer = None

#############################################################
#implementing a text processing function

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
######Process text function
def preprocess_text(text):
    text = text.lower()#to lowercase text
    text = re.sub(r'<.*?>', '', text)#remove html tags if any
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)#remove special symbol and puntuation
    tokens = text.split()#spliting text into list of individual
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)
#------------Flask application instance--------------------------
app = Flask(__name__)
@app.route('/')
def home():
    return jsonify({'message': 'Sentiment Analysis API is live! Use POST /predict with JSON {"review": "your text"}'})

#------------define prediction end point-------------------------
@app.route('/predict', methods=['POST'])##post method is used
def predict():
    ##get JSON data from POST request method
    try:
        data = request .get_json()
        if data is None:
            return jsonify({'error':'Invalid input no json data recived'})
    except Exception as e:
        return jsonify({'error': f'An error occurred while parsing JSON: {str(e)}'}), 400
    ##Extract review text from json data
    review_text = data.get('review')
    if not review_text or not isinstance(review_text, str) or not review_text.strip():
        # Return a specific, helpful error message and a 400 Bad Request status code.
        return jsonify({'error': 'The "review" field is missing, empty, or not a string.'}), 400
    ####preprocess this review text----------
    cleaned_review = preprocess_text(review_text)
    #-----------VEctorize the cleaned review using fit and transform----------------
    try:
        text_vector = vectorizer.transform([cleaned_review])
    except Exception as e:
        return jsonify({'error': f'An error occurred during text vectorization: {str(e)}'}), 500

    #---------------use the loaded model to perform prediction---------------------------
    try:
        prediction = model.predict(text_vector)
        sentiment = prediction[0]
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500
    #---------return json format-0-------------------
    return jsonify({'sentiment': sentiment.title()}), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






















