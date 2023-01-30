import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from io import StringIO
import time

nltk.download('punkt')
nltk.download('stopwords')

st.header(':blue[Text Analysis] with _:green[NLTK], :green[Vader], and :green[TextBlob]_')

st.subheader('Only upload a file with .txt format')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()

    with st.spinner('Wait for it...'):
        time.sleep(3)

    def vader_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        return sentiment


    def textblob_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment


    def most_common_words(text, num=10):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha() == True]
        fdist = FreqDist(filtered_tokens)
        return fdist.most_common(num)


    col1, col2 = st.columns(2)
    with col1:
        st.success('Vader Sentiment')
        st.info(f"Negative: {round(vader_sentiment(string_data)['neg'], 2)} -- Positive: {round(vader_sentiment(string_data)['pos'], 2)}")
        st.info(f"Neutral: {round(vader_sentiment(string_data)['neu'], 2)} -- Total: {round(vader_sentiment(string_data)['compound'], 2)}")

    with col2:
        st.success('TextBlob Sentiment')
        st.info(f"Polarity: {round(textblob_sentiment(string_data).polarity, 2)}")
        st.info(f"Subjectivity: {round(textblob_sentiment(string_data).subjectivity, 2)}")

    st.success('10 common words')
    st.info(most_common_words(string_data))
