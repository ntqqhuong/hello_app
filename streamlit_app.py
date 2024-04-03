import streamlit as st
import pickle

# Load mô hình từ tệp .pkl
with open('model_cnn_bilstm.pkl', 'rb') as f:
    model = pickle.load(f)

# create title
st.title('Predicting sentiment ')
review = st.text_input('Enter a review ')

submit = st.button('Predict')

if submit:
    # Chắc chắn rằng đánh giá được chuyển đổi thành định dạng phù hợp cho mô hình
    input_data = [review]

    # Thực hiện dự đoán
    prediction = model.predict(input_data)

    # Hiển thị dự đoán
    st.write(prediction)
