import streamlit as st
from tensorflow.keras.models import load_model

# Tải lại mô hình đã được huấn luyện
model = load_model('model_cnn_bilstm.h5')

# Tạo tiêu đề
st.title('Đánh giá trải nghiệm người dùng app Viettel')
review = st.text_input('Nhập đánh giá của bạn')

submit = st.button('Dự đoán')

if submit:
    # Dự đoán với mô hình
    prediction = model.predict([review])

    # Hiển thị dự đoán
    st.write(prediction)
