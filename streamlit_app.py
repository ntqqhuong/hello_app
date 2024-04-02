import streamlit as st
import pickle

# loading the trained model
model = pickle.load(open('model_cnn_bilstm.h5', 'rb'))

# create title
st.title('Đánh giá trải nghiệm người dùng app Viettel ')
review= st.text_input('Enter a review ')

submit = st.button('Predict')

if submit:
    prediction = model.predict([review])

    # print(prediction)
    # st.write(prediction)
  
