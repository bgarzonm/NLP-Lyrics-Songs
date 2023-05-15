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


# Find the songs with the highest and lowest sentiment scores
highest_sentiment = df.loc[df["sentiment"].idxmax()]
lowest_sentiment = df.loc[df["sentiment"].idxmin()]

print("Song with highest sentiment score: ")
print(highest_sentiment["title"])
print(highest_sentiment["lyrics"])
print("Sentiment score: ", highest_sentiment["sentiment"])

print("Song with lowest sentiment score: ")
print(lowest_sentiment["title"])
print(lowest_sentiment["lyrics"])
print("Sentiment score: ", lowest_sentiment["sentiment"])

# Analyze the most frequent words in the lyrics
from collections import Counter

all_words = [word for song in df["tokens"] for word in song]
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

plt.bar([word[0] for word in common_words], [word[1] for word in common_words])
plt.xticks(rotation=45)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Convert the word vectors to a matrix
word_vectors = np.array([model.wv[word] for word in word_counts.keys()])

# Perform dimensionality reduction with PCA
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Aggregate sentiment values by word
sentiments_by_word = {}
for word in word_counts.keys():
    word_sentiments = [df.loc[df["lyrics"].str.contains(word), "sentiment"].mean()]
    sentiments_by_word[word] = word_sentiments

# Convert sentiment dictionary to an array
sentiment_array = np.array([sentiments_by_word[word][0] for word in word_counts.keys()])

# Cluster the songs based on their sentiment and word usage
X = np.column_stack((sentiment_array, word_vectors_2d))

# Replace NaNs with mean of column
nan_cols = np.isnan(X)
col_means = np.nanmean(X, axis=0)
X[nan_cols] = np.take(col_means, np.where(nan_cols)[1])

# Cluster the songs based on their sentiment and word usage
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)


# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["red", "green", "blue", "purple"]
for i in range(len(X)):
    plt.scatter(X[i, 1], X[i, 2], color=colors[kmeans.labels_[i]])
plt.xlabel("Sentiment Score")
plt.ylabel("Word Vector (PCA)")
plt.show()

fig.savefig("../visualization/cluster_plot.png")
