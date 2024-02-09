import sys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

samples = ["The service was not the best because the waiting time was so long",
           "In my opinion, the service was excellent",
           "I had to wait for a long time. Not recommended",
           "I would want to go again"]

samples_positive = []
samples_negative = []
samples_neutral = []

for sample in samples:
    sentiment_scores = sia.polarity_scores(sample)
    if sentiment_scores['compound'] >= 0.05:
        samples_positive.append(sample)
    elif sentiment_scores['compound'] <= -0.05:
        samples_negative.append(sample)
    else:
        samples_neutral.append(sample)


words = ' '.join(samples_negative).split()
word_counts = Counter(words)
ranked_words = word_counts.most_common()

for word, count in ranked_words:
    print(f"{word}: {count}")





