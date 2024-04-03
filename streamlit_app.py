import streamlit as st
from tensorflow.keras.models import load_model

# loading the trained model from .h5 file
model = load_model('model_cnn_bilstm.h5')

# create title
st.title('Predicting sentiment')
review = st.text_input('Enter a review')

submit = st.button('Predict')

if submit:
    # Reshape the input review to fit the model's input shape
    review = [review]  # Convert the review to a list
    prediction = model.predict(review)

    print(prediction)
    st.write(prediction)

