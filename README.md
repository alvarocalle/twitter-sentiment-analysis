# Twitter Sentiment Analysis

Set of functionalities to store and analyze tweets using the Twitter API in Python. Tweets are retrieved and stored in a MongoDB, SQLittle or text file.

## Twitter API
To run this application, it is needed to have the twitter API up. To set up the API do the following:

1. Create a twitter account
2. Go to https://developer.twitter.com/en/apps
3. Create an New Application hitting the "Create an app" button and go to the needed steps
4. Once the app has been created copy the keys and tokens (API keys, API secret key, Access token and Access token secret)
5. These keys and tokens are needed to be able to use the API and are stored in env.vars file.
6. It is good to have these stored as environment variables

## Configuration
1. The different paths and options needed by the APP can be configured in twitter.conf. Some options are:
- SENTIMENT:
    - SENTIMENT='lexicon' uses the lexicon method (only Spanish) to do sentiment analysis. This methid does not need Google Cloud API running and it's the default option. If we want to change the lexicon to another language we need to replace the files data/negativas\_mejorada.csv y data/positivas\_mejorada.csv.
    - SENTIMENT='api' uses Google Cloud API for sentiment analysis.
    - SENTIMENT=None, does not carry out sentiment analysis.
- TRACK:
    - Hashtag or quote to filter tweets: Example TRACK='@metro\_madrid' filters tweets related to Metro Madrid.
-cols: columns that will be extracted from the tweet for procesing.

## APP
- twitter.py : libraries and classes
- twitter_historical.py : data collection and storage
- twitter_wordcloud.py : WordCloud

## Dependences
- requirements.txt

## Authors

* **Alvaro Calle Cordon**
