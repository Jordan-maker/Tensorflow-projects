import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

# download nltk corpus (first time only)
#nltk.download('all')

"""
This template make sentimental predictions from opinions extracted from Amazon.
An additional text preprocessing is made in order to determine if can help to increase the accuracy in the prediction, 
due to SentimentIntensityAnalyzer() already perform internally some pre-processing.
"""

# Load the amazon review dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')#, chunksize=100)

# vader_lexicon
analyzer = SentimentIntensityAnalyzer()

# create preprocess_text function
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens=[]
    for token in tokens:
        if token not in stopwords.words('english'):
            filtered_tokens.append(token)

    # Lemmatize the tokens
    lemmatized_tokens=[]
    lemmatizer = WordNetLemmatizer()
    for token in filtered_tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def predict_sentiment(text):

    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:      #positive
        return 1
    elif sentiment_score['compound'] <= -0.05:  #negative
        return 0
    else:
        return -1


# Add a new column into the original dataframe with the processed text
df['reviewText_modified'] = df['reviewText'].apply(preprocess_text)

# make prediction on the processed text using vader_lexicon analyzer.
# v1: using the text chain modified.
# v2: using the raw text

df['Positive_prediction_v1'] = df['reviewText_modified'].apply(predict_sentiment)
df['Positive_prediction_v2'] = df['reviewText'].apply(predict_sentiment)

successes_v1 = sum(df['Positive_prediction_v1'] == df['Positive'])
successes_v2 = sum(df['Positive_prediction_v2'] == df['Positive'])

accuracy_v1 = successes_v1/len(df)
accuracy_v2 = successes_v2/len(df)

print(accuracy_v1, accuracy_v2)

# From these results (0.7879, 0.8), we can observe that apply this analyzer could be
# better on the raw text. Is not necessary make the preprocess defined by the function preprocess_text()
# because internally SentimentIntensityAnalyzer() tokenizes the input text, removes punctuation and special characters,
# converts words to lowercase and then assigns sentiment scores.






