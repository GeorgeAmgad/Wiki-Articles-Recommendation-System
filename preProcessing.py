import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import globals

# all are less than 700 article
# If not previously performed:
# nltk.download('punkt')
# nltk.download('stopwords')

stemming = PorterStemmer()
stops = set(stopwords.words("english"))


def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X


def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""

    # Convert to lower case
    text = raw_text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]

    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]

    # Remove stop words
    meaningful_words = [w for w in stemmed_words if w not in stops]

    # Rejoin meaningful stemmed words
    joined_words = (" ".join(meaningful_words))

    # Return cleaned data
    return joined_words


### APPLY FUNCTIONS TO EXAMPLE DATA


# Load data example

articles = pd.read_csv('train.csv')

# Truncate data for example
articles = articles.head(100)  # number of articles to use , we use 100 out of about 700
# imdb = imdb["articles"]
# print(imdb)

# Get text to clean
text_to_clean = list(articles['articles'])
globals.original_text_train = text_to_clean.copy()

# Clean text
cleaned_text = apply_cleaning_function_to_list(text_to_clean)
globals.cleaned_text_train = cleaned_text.copy()

# Show first example
# print('Original text:', text_to_clean[0])
# print('\nCleaned text:', cleaned_text[0])

# Add cleaned data back into DataFrame
articles['cleaned_review'] = cleaned_text
