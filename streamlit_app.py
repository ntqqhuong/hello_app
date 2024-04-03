import streamlit as st
# loading the trained model
model = pickle.load(open('model_cnn_bilstm.pickle', 'rb'))
# Tạo tiêu đề
st.title('Đánh giá trải nghiệm người dùng app Viettel')
review = st.text_input('Nhập đánh giá của bạn')
submit = st.button('Predict')
if submit:
    # Dự đoán với mô hình
    prediction = model.predict([review])

    # Hiển thị dự đoán
    st.write(prediction)
