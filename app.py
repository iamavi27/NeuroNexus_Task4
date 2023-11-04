import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # <!-- lower case -->
    text = nltk.word_tokenize(text)  # <!-- tokeniation -->

    y = []
    for i in text:  # <!-- removing special char -->
        if i.isalnum():
            y.append(i)

    text = y[:]  # for copy the data from 1 var to another do cloning
    y.clear()

    for i in text:
        if i not in stopwords.words(
                'english') and i not in string.punctuation:  # <!-- removing stop words and punctuation -->
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # steaming ->  removing affixes from a word so that we are left with the stem of that word
        # like run, rans, running all convert to run

    return " ".join(y)


tfidf = pickle.load(open('vectotizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS SPAM DETECTION")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vectorized_input = tfidf.transform([transformed_sms])

    result = model.predict(vectorized_input)[0]

    if result == 1:
        st.header("Spam Messages")
    else:
        st.header("Not Spam Messages")