import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Sample data: Replace with your own CSV or data source
data = {
    'tweet': [
        'I love the new design of your website! 😍',
        'Worst customer service ever. Totally disappointed.',
        'The product is okay, but shipping was slow.',
        'Absolutely fantastic experience, will buy again!',
        'Not worth the price. Very poor quality.',
        'Great support team, helped me quickly.',
        'I am not happy with the purchase.',
        'Best purchase I have made this year!',
        'Terrible, will not recommend to anyone.',
        'It works as expected. Satisfied.'
    ]
}
df = pd.DataFrame(data)
df.head()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)
df.head()

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_tweet'].apply(get_sentiment)
df[['tweet', 'sentiment']]

# Sentiment distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', hue='sentiment', data=df, palette='Set2', legend=False)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Show example tweets for each sentiment
for sentiment in ['Positive', 'Negative', 'Neutral']:
    print(f'\nExample {sentiment} tweets:')
    print(df[df['sentiment'] == sentiment]['tweet'].head(2).to_string(index=False))

