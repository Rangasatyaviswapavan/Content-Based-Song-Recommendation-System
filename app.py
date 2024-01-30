from flask import Flask, render_template, request
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv('spotify_millsongdata.csv')
df.drop_duplicates(inplace=True)
df.drop('link', axis=1, inplace=True)
df['lyrics'] = df['lyrics'].str.lower()
df['lyrics'] = df['lyrics'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['lyrics'] = df['lyrics'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Tokenization and stemming function
stemmer = PorterStemmer()
def tokenize_and_stem(lyrics):
    tokens = word_tokenize(lyrics)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Vectorize lyrics using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english', max_features=10000)
lyrics_features = vectorizer.fit_transform(df['lyrics'])
similarity_matrix = cosine_similarity(lyrics_features, lyrics_features)

# Recommendation function
def recommendation(song_name):
    dist = sorted(list(enumerate(similarity_matrix[df[df['song'] == song_name].index[0]])), reverse=True, key=lambda x: x[1])
    res = []
    for i in dist[1:9]:
        res.append(df.iloc[i[0]].song)
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_songs = []
    if request.method == 'POST':
        song_name = request.form['song']
        recommended_songs = recommendation(song_name)
    return render_template('index.html', recommended_songs=recommended_songs)

if __name__ == '__main__':
    app.run(debug=True)
