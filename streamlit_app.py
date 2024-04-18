import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer

# Load tokenizer function
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Load model function
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess raw input function
def preprocess_raw_input(raw_input, tokenizer):
    input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(raw_input))
    input_text_pre = " ".join(input_text_pre)
    input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=120)
    return vec_data

# Inference model function
def inference_model(input_feature, model):
    output = model(input_feature).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {'Tiêu cực':0, 'Tích cực':1}
    label = list(label_dict.keys())
    return label[int(result)], conf

# Prediction function
def prediction(raw_input, tokenizer, model):
    input_model = preprocess_raw_input(raw_input, tokenizer)
    result, conf = inference_model(input_model, model)
    return result, conf

# Load model and tokenizer
my_model = load_model('model_cnn_bilstm.h5')
my_tokenizer = load_tokenizer("tokenizer_data.pkl")

# Streamlit UI
st.title("Sentiment Analysis")

input_text = st.text_area("Enter the text")

if st.button('Predict'):
    result, conf = prediction(input_text, my_tokenizer, my_model)
    sentiment_label = "Negative" if result == 0 else "Positive"
    st.write(f"Predicted sentiment: {sentiment_label}, Confidence: {conf}")
