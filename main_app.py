import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the imdb dataset and perform preprocessing
word_index = imdb.get_word_index()

reverse_word_index = {value: key for (key, value) in word_index.items()}

#load the pre-trained model
model = load_model('imdb_classification.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3  for word in words]
    padded = pad_sequences([encoded], maxlen=500, padding='pre')
    return padded

### prediction function 
def predict_sentiments(review):
    preprocessed_input = preprocess_text(review)
    
    predict = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if predict[0][0] > 0.5 else 'Negative'
    
    return sentiment, predict[0][0]

##streanlit app 
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the prediction result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Probability: {prediction[0][0]:.2f}")
else :
    st.write("Please enter a movie review.")