# %%
import openai
import tensorflow
import sys
import newsapi
from newsapi.newsapi_client import NewsApiClient

# %%
# Initialize News API and OpenAI
newsapi = NewsApiClient(api_key='f09fcd4ec096415f92e5be805a248f70')
openai_api_key = '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'

# %%
# Fetch news articles
def fetch_news(stock_symbol, industry, date):
    all_articles = newsapi.get_everything(
        q=f"{stock_symbol} AND {industry}",
                                          from_param=date,
                                          to=date,
                                          language='en',
        sort_by='publishedAt'
    )
    return [article['content'] for article in all_articles['articles'] if article['content']]

# Sentiment analysis with OpenAI
def analyze_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant skilled in sentiment analysis."},
            {"role": "user", "content": f"Analyze the sentiment of this text and then after output either 'Positive', 'Negative', or 'Neutral'. Do not output anything other than one of those words: {text}"}
        ],
        api_key=openai_api_key  # Ensure your API key is correctly set here
    )
    return response.choices[0].message['content'].strip()

# %%
# Main function to analyze stock sentiment
def analyze_stock_sentiment(stock_symbol, industry, date):
    articles = fetch_news(stock_symbol, industry, date)
    
    positive, negative, neutral = 0, 0, 0
    sentiments = [analyze_sentiment(article) for article in articles]
    
    for sentiment in sentiments:
        if sentiment == "Positive":
            positive += 1
        elif sentiment == "Negative":
            negative += 1
        else:
            neutral += 1
    
    print(positive, negative, neutral)
    
    # Determine overall sentiment
    total = positive + negative + neutral
    if total > 0:
        score = ((positive - negative) / total) * 100
    else:
        score = 0  # Handle the case where there are no sentiments at all
        
    return score

# %%
# ... remaining code without Keras-RL dependencies ...







