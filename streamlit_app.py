import requests
import re
import streamlit as st
import pickle



model = pickle.load(open('model_cnn_bilstm.pkl', 'rb'))


st.title("Predict Sentiment of Movie Review")
st.markdown(
"""
1. I proceeded to build a sentiment classification model for IMDB reviews using TF-IDF.
2. Utilizing different classification algorithms, the LinearSVC model achieved the highest accuracy of 91.23%.
3. Then, I employed pickle and streamlit to develop a web application that predicts emotions for any review.
""")

review = st.text_area('Please enter your review 👇👇', key='review_input')
st.markdown("""
    <style>
        label[for="review_input"]::before {
            content: "Please enter your review ";
            font-size: 40px !important;
        }
    </style>
""", unsafe_allow_html=True)


submit = st.button('Predict')      
if submit:
    prediction = model.predict([review])

    if prediction[0] == 'positive':
        st.success('Positive Review')
    else:
        st.warning('Negative Review')
        
st.markdown(
    """
    <style>
    body {
        background-color: #ff1493;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Introduce | IMDb ✨✨✨")
st.markdown(
"""
1. IMDb (Internet Movie Database) is an online database of information related to films, television series, podcasts, home videos, video games,
and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews. 
IMDb began as a fan-operated movie database on the Usenet group "rec.arts.movies" in 1990, and moved to the Web in 1993. 
Since 1998, it has been owned and operated by IMDb.com, Inc., a subsidiary of Amazon.

2. As of 2019, IMDb was the 52nd most visited website on the Internet, as ranked by Alexa. 
As of March 2022, the database contained some 10.1 million titles (including television episodes), 11.5 million person records, and 83 million registered users.
"""
)
