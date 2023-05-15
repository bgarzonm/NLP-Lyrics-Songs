import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the song lyrics into a pandas dataframe
df = pd.read_csv("../lyrics/drake.csv")

# Preprocess the lyrics
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))


def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


df["tokens"] = df["lyrics"].apply(preprocess)

# Train a Word2Vec model on the lyrics
model = Word2Vec(df["tokens"], min_count=1)

# Define a function to calculate the sentiment score of a sentence
sid = SentimentIntensityAnalyzer()


def get_sentiment_score(sentence):
    return sid.polarity_scores(sentence)["compound"]


# Define a function to calculate the sentiment score of a song
def get_song_sentiment_score(song_tokens):
    song_vector = np.mean(
        [model.wv[token] for token in song_tokens if token in model.wv], axis=0
    )
    song_text = " ".join(song_tokens)
    song_sentiment_score = get_sentiment_score(song_text)
    return song_sentiment_score


# Calculate the sentiment score of each song
df["sentiment"] = df["tokens"].apply(get_song_sentiment_score)

# Results
df[["title", "sentiment"]][:5]


# Plot a scatter plot of sentiment score vs. length of lyrics

df["lyrics_length"] = df["lyrics"].apply(len)
plt.scatter(df["sentiment"], df["lyrics_length"])
plt.xlabel("Sentiment Score")
plt.ylabel("Length of Lyrics")
plt.show()
