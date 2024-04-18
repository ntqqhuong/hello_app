import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    
tfidf = pickle.load(open('tokenizer_data.pkl','rb'))
model = pickleload(open('model_cnn_bilstm.pkl','rb'))

st.title("Sentiment Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Positive")
    elif result==0:
        st.header("Negative")
