# Content-Based Song Recommendation System

This project implements a simple content-based recommendation system for songs using Python and Flask. It analyzes song lyrics to recommend similar songs based on their textual content.

## Features

- Tokenization and stemming of song lyrics
- TF-IDF vectorization for feature extraction
- Cosine similarity calculation for song similarity
- Basic front end using Flask for user interaction

## Usage

1. Install the required Python libraries:
    ```bash
    pip install Flask pandas scikit-learn nltk
    ```
2. Clone the repository:
    ```bash
    git clone https://github.com/Rangasatyaviswapavan/Content-Based-Song-Recommendation-System.git
    ```
3. Navigate to the project directory:
    ```bash
    cd content-based-song-recommendation
    ```
4. Run the Flask app:
    ```bash
    python app.py
    ```
5. Open your web browser and go to `http://localhost:5000` to access the recommendation system.

## Dataset

The dataset used for this project is available in the file `spotify_millsongdata.csv`. It contains song names and their corresponding lyrics.

## Credits

This project was created as a practice exercise for learning about content-based recommendation systems. It utilizes the following libraries and tools:

- Flask: Web framework for building the front end
- NLTK: Natural Language Toolkit for text preprocessing
- scikit-learn: Machine learning library for TF-IDF vectorization and cosine similarity calculation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
