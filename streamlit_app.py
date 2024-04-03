import streamlit as st
import pickle

# loading the trained model
model = pickle.load(open('model_cnn_bilstm.pkl', 'rb'))

# create title
st.title('Predicting sentiment ')
review= st.text_input('Enter a review ')

submit = st.button('Predict')

if submit:
    prediction = model.predict([review])

    print(prediction)
    st.write(prediction)
